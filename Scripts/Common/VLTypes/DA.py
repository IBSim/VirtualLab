import os
import sys
sys.dont_write_bytecode=True
from types import SimpleNamespace as Namespace
from importlib import import_module
import Scripts.Common.VLFunctions as VLF
import copy
import pickle

'''
DA - Data Analysis
'''

def Setup(VL, **kwargs):
    VL.SIM_DA = "{}/DA".format(VL.SIM_SCRIPTS)
    VL.DAData = {}
    DADicts = VL.CreateParameters(VL.Parameters_Master, VL.Parameters_Var,'DA')

    # if either DADicts is empty or RunDA is False we will return
    if not (kwargs.get('RunDA', True) and DADicts): return

    VL.tmpDA_DIR = "{}/DA".format(VL.TEMP_DIR)
    os.makedirs(VL.tmpDA_DIR, exist_ok=True)

    for DAName, ParaDict in DADicts.items():
        CALC_DIR = "{}/{}".format(VL.PROJECT_DIR, DAName)
        DADict = {'Name':DAName,
                 'CALC_DIR':CALC_DIR,
                 'TMP_CALC_DIR':"{}/{}".format(VL.tmpDA_DIR, DAName),
                 'Parameters':Namespace(**ParaDict),
                 'Data':{}}

        # Important information can be added to Data during any stage of the
        # data analysis, and this will be saved to the location specified by the
        # value for the __file__ key
        DADict['Data'] = {'__file__':"{}/Data.pkl".format(DADict['CALC_DIR'])}

        if VL.mode in ('Headless','Continuous'):
            DADict['LogFile'] = "{}/Output.log".format(DADict['CALC_DIR'])
        else : DADict['LogFile'] = None

        os.makedirs(CALC_DIR, exist_ok=True)
        os.makedirs(DADict["TMP_CALC_DIR"],exist_ok=True)


        VL.DAData[DAName] = DADict

def PoolRun(VL, DADict):
    Parameters = DADict["Parameters"]
    VLF.WriteData("{}/Parameters.py".format(DADict['CALC_DIR']), Parameters)

    DAmod = import_module(Parameters.File)
    DASgl = getattr(DAmod, 'Single', None)
    err = DASgl(VL,DADict)
    return err

def Run(VL,**kwargs):
    if not VL.DAData: return
    sys.path.insert(0,VL.SIM_DA)

    NumThreads = kwargs.get('NumThreads',1)
    launcher = kwargs.get('launcher','Process')

    VL.Logger('\n### Starting Data Analysis ###\n', Print=True)

    NbDA = len(VL.DAData)
    DADicts = list(VL.DAData.values())
    PoolArgs = [[VL]*NbDA,DADicts]

    N = min(NumThreads,NbDA)

    if launcher == 'Sequential':
        Res = []
        for args in zip(*PoolArgs):
            ret = VLF.VLPool(PoolRun,*args)
            Res.append(ret)
    elif launcher == 'Process':
        from pathos.multiprocessing import ProcessPool
        pool = ProcessPool(nodes=N, workdir=VL.TEMP_DIR)
        Res = pool.map(VLF.VLPool,[PoolRun]*NbDA, *PoolArgs)
    elif launcher == 'MPI':
        from pyina.launchers import MpiPool
        # Ensure that all paths added to sys.path are visible in pyinas MPI subprocess
        addpath = set(sys.path) - set(VL._pypath) # group subtraction
        addpath = ":".join(addpath) # write in unix style
        PyPath_orig = os.environ.get('PYTHONPATH',"")
        os.environ["PYTHONPATH"] = "{}:{}".format(addpath,PyPath_orig)

        onall = kwargs.get('onall',True) # Do we want 1 mpi worker to delegate and not compute (False if so)
        if not onall and NumThreads > N: N=N+1 # Add 1 if extra threads available for 'delegator'

        pool = MpiPool(nodes=N,source=True, workdir=VL.TEMP_DIR)
        Res = pool.map(VLF.VLPool,[PoolRun]*NbDA, *PoolArgs, onall=onall)

        # reset environment back to original
        os.environ["PYTHONPATH"] = PyPath_orig

    Errorfnc = VLF.VLPoolReturn(DADicts,Res)
    if Errorfnc:
        VL.Exit("The following DA routine(s) finished with errors:\n{}".format(Errorfnc))

    VL.Logger('\n### Data Analysis Complete ###',Print=True)
