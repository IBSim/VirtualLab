
import os
import sys
sys.dont_write_bytecode=True
from types import SimpleNamespace as Namespace
from importlib import import_module
from Scripts.Common.VLFunctions import VLPool, VLPoolReturn
import copy

def Setup(VL, **kwargs):
    VL.MLData = {}
    MLDicts = VL.CreateParameters(VL.Parameters_Master, VL.Parameters_Var,'ML')

    # if either MLDicts is empty or RunML is False we will return
    if not (kwargs.get('RunML', True) and MLDicts): return

    VL.tmpML_DIR = "{}/ML".format(VL.TEMP_DIR)
    os.makedirs(VL.tmpML_DIR, exist_ok=True)

    VL.ML_DIR = "{}/ML".format(VL.PROJECT_DIR)
    os.makedirs(VL.ML_DIR, exist_ok=True)

    for MLName, ParaDict in MLDicts.items():
        if 'SubDir' in ParaDict:
            CALC_DIR = "{}/{}/{}".format(VL.ML_DIR, ParaDict["SubDir"], MLName)
        else:
            CALC_DIR = "{}/{}".format(VL.ML_DIR, MLName)
        MLDict = {'Name':MLName,
                 'CALC_DIR':CALC_DIR,
                 'TMP_CALC_DIR':"{}/{}".format(VL.tmpML_DIR, MLName),
                 'Parameters':Namespace(**ParaDict)}

        os.makedirs(MLDict["CALC_DIR"], exist_ok=True)
        os.makedirs(MLDict["TMP_CALC_DIR"])
        if VL.mode in ('Headless','Continuous'):
            MLDict['LogFile'] = "{}/Output.log".format(MLDict['CALC_DIR'])
        else : MLDict['LogFile'] = None

        VL.WriteModule("{}/Parameters.py".format(MLDict['CALC_DIR']), ParaDict)

        VL.MLData[MLName] = MLDict

def PoolRun(VL, MLDict):
    Parameters = MLDict["Parameters"]

    MLmod = import_module(Parameters.File)
    MLSgl = getattr(MLmod, 'Single', None)
    err = MLSgl(VL,MLDict)
    return err

def devRun(VL,**kwargs):
    if not VL.MLData: return
    sys.path.insert(0,VL.SIM_ML)

    NumThreads = kwargs.get('NumThreads',1)

    VL.Logger('\n### Starting Machine Learning ###\n', Print=True)

    # Run high throughput part in parallel
    NbML = len(VL.MLData)
    MLDicts = list(VL.MLData.values())
    PoolArgs = [[VL]*NbML,MLDicts]

    launcher = kwargs.get('launcher','Process')
    Res = []
    if launcher == 'Sequential':
        for args in zip(*PoolArgs):
            ret = VLPool(PoolRun,*args)
            Res.append(ret)
    elif launcher == 'Process':
        from pathos.multiprocessing import ProcessPool
        pool = ProcessPool(nodes=NumThreads, workdir=VL.TEMP_DIR)
        Res = pool.map(VLPool,[PoolRun]*NbML, *PoolArgs)
    elif launcher == 'MPI':
        from pyina.launchers import MpiPool
        # Ensure that all paths added to sys.path are visible pyinas MPI subprocess
        addpath = set(sys.path) - set(VL._pypath) # group subtraction
        addpath = ":".join(addpath) # write in unix style
        PyPath_orig = os.environ.get('PYTHONPATH',"")
        os.environ["PYTHONPATH"] = "{}:{}".format(addpath,PyPath_orig)

        onall = kwargs.get('onall',True) # Do we want 1 mpi worked to delegate and not compute (False if so)
        pool = MpiPool(nodes=NumThreads,source=True, workdir=VL.TEMP_DIR)
        # TryPathos gives a try and except block around the function to prevent
        # hanging which can occur with mpi4py
        Res = pool.map(VLPool,[PoolRun]*NbML, *PoolArgs, onall=onall)

        # reset environment back to original
        os.environ["PYTHONPATH"] = PyPath_orig

    Errorfnc = VLPoolReturn(MLDicts,Res)
    if Errorfnc:
        VL.Exit("The following ML routine(s) finished with errors:\n{}".format(Errorfnc))

    MLmod = import_module(VL.Parameters_Master.ML.File)
    if hasattr(MLmod,'Combined'):
        MLmod.Combined(VL,VL.MLData.values())

    VL.Logger('\n### Machine Learning Complete ###',Print=True)
