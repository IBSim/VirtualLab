
import os
import sys
sys.dont_write_bytecode=True
from types import SimpleNamespace as Namespace
from importlib import import_module
from contextlib import redirect_stderr, redirect_stdout
from pathos.multiprocessing import ProcessPool
import copy
# from Scripts.Common.VLPackages import CodeAster
from ..VLPackages import CodeAster

def CheckFile(Directory,fname,ext):
    if not fname:
        return True
    else:
        return os.path.isfile('{}/{}.{}'.format(Directory,fname,ext))

def Setup(VL,**kwargs):

    os.makedirs(VL.STUDY_DIR, exist_ok=True)
    VL.SimData = {}

    if not kwargs.get('RunSim', True): return

    MetaInfo = {key:val for key,val in VL.__dict__.items() if type(val)==str}

    SimDicts = VL.CreateParameters(VL.Parameters_Master, VL.Parameters_Var,'Sim')
    for SimName, ParaDict in SimDicts.items():
        # Run checks
        # Check files exist
        if not CheckFile(VL.SIM_PREASTER,ParaDict.get('PreAsterFile'),'py'):
        	VL.Exit("PreAsterFile '{}.py' not in directory {}".format(ParaDict['PreAsterFile'],VL.SIM_PREASTER))
        if not CheckFile(VL.SIM_ASTER,ParaDict.get('AsterFile'),'comm'):
        	VL.Exit("AsterFile '{}.comm' not in directory {}".format(ParaDict['AsterFile'],VL.SIM_ASTER,))
        if not CheckFile(VL.SIM_POSTASTER, ParaDict.get('PostAsterFile'), 'py'):
        	VL.Exit("PostAsterFile '{}.py' not in directory {}".format(ParaDict['PostAsterFile'],VL.SIM_POSTASTER))
        # Check mesh will be available
        if not (ParaDict['Mesh'] in VL.MeshData or CheckFile(VL.MESH_DIR, ParaDict['Mesh'], 'med')):
        	VL.Exit("Mesh '{}' isn't being created and is not in the mesh directory '{}'".format(ParaDict['Mesh'], VL.MESH_DIR))
        # Check materials used
        Materials = ParaDict.get('Materials',[])
        if type(Materials)==str: Materials = [Materials]
        elif type(Materials)==dict: Materials = Materials.values()
        MatErr = [mat for mat in set(Materials) if not os.path.isdir('{}/{}'.format(VL.MATERIAL_DIR, mat))]
        if MatErr:
        		VL.Exit("Material(s) {} specified for {} not available.\n"\
        		"Please see the materials directory {} for options.".format(MatErr,SimName,VL.MATERIAL_DIR))
        # Checks complete

        # Create dict of simulation specific information to be nested in SimData
        StudyDict = {}
        StudyDict['TMP_CALC_DIR'] = TMP_CALC_DIR = "{}/{}".format(VL.TMP_DIR, SimName)
        StudyDict['CALC_DIR'] = CALC_DIR = "{}/{}".format(VL.STUDY_DIR, SimName)
        StudyDict['PREASTER'] = "{}/PreAster".format(CALC_DIR)
        StudyDict['ASTER'] = "{}/Aster".format(CALC_DIR)
        StudyDict['POSTASTER'] = "{}/PostAster".format(CALC_DIR)
        StudyDict['MeshFile'] = "{}/{}.med".format(VL.MESH_DIR, ParaDict['Mesh'])

        # Create tmp directory & add __init__ file so that it can be treated as a package
        if not os.path.isdir(TMP_CALC_DIR): os.makedirs(TMP_CALC_DIR)
        with open("{}/__init__.py".format(TMP_CALC_DIR),'w') as f: pass
        # Combine Meta information with that from Study dict and write to file for salome/CodeAster to import
        VL.WriteModule("{}/PathVL.py".format(TMP_CALC_DIR), {**MetaInfo, **StudyDict})
        # Write Sim Parameters to file for Salome/CodeAster to import
        VL.WriteModule("{}/Parameters.py".format(TMP_CALC_DIR), ParaDict)

        # Attach Parameters to StudyDict for ease of access
        StudyDict['Parameters'] = Namespace(**ParaDict)
        # Add StudyDict to SimData dictionary
        VL.SimData[SimName] = StudyDict.copy()

def PoolRun(VL, StudyDict, kwargs):
    RunPreAster = kwargs.get('RunPreAster',True)
    RunAster = kwargs.get('RunAster', True)
    RunPostAster = kwargs.get('RunPostAster', True)

    Parameters = StudyDict["Parameters"]

    if VL.mode == 'Headless':OutFile = "{}/Output.log".format(StudyDict['TMP_CALC_DIR'])
    elif VL.mode == 'Continuous':OutFile = "{}/Output.log".format(StudyDict['TMP_CALC_DIR'])
    else : OutFile=''

    # Copy original dictionary so that we can tell if anything gets added to it
    OrigDict = copy.deepcopy(StudyDict)
    Error=None

    # Create a blank file
    if OutFile:
        with open(OutFile,'w') as f: pass

    if RunPreAster and hasattr(Parameters,'PreAsterFile'):
        sys.path.insert(0, VL.SIM_PREASTER)
        PreAster = import_module(Parameters.PreAsterFile)
        PreAsterSgl = getattr(PreAster, 'Single',None)

        VL.Logger("Running PreAster for '{}'\n".format(Parameters.Name),Print=True)
        os.makedirs(StudyDict['PREASTER'],exist_ok=True)

        if not OutFile:
            err = PreAsterSgl(VL,StudyDict)
        else :
            with open(OutFile,'a') as f:
                with redirect_stdout(f), redirect_stderr(f):
                    err = PreAsterSgl(VL,StudyDict)

        if err:
            Error = ['PreAster',err]
            RunAster=RunPostAster=False

    if RunAster and hasattr(Parameters,'AsterFile'):
        VL.Logger("Running Aster for '{}'\n".format(Parameters.Name),Print=True)

        os.makedirs(StudyDict['ASTER'],exist_ok=True)
        # Create export file for CodeAster
        ExportFile = "{}/Export".format(StudyDict['ASTER'])
        CommFile = '{}/{}.comm'.format(VL.SIM_ASTER, Parameters.AsterFile)
        MessFile = '{}/AsterLog'.format(StudyDict['ASTER'])
        VL.CodeAster.ExportWriter(ExportFile, CommFile,
        							StudyDict["MeshFile"],
        							StudyDict['ASTER'],
                                    MessFile, **kwargs)

        if VL.mode == 'Headless': AsterOut='/dev/null'
        elif VL.mode == 'Continuous': AsterOut = OutFile.format(StudyDict['ASTER'])
        else : AsterOut=''

        SubProc = VL.CodeAster.Run(ExportFile, Name=Parameters.Name,
                                     AddPath=[VL.TMP_DIR,StudyDict['TMP_CALC_DIR']],
                                     OutFile=AsterOut)
        err = SubProc.wait()
        if err:
            Error = ['Aster',"Aster SubProc returned code {}".format(err)]
            RunPostAster=False

        # from subprocess import Popen
        # SubProc = Popen(['echo','Hello World'])
        # SubProc.wait()

    if RunPostAster and hasattr(Parameters,'PostAsterFile'):
        sys.path.insert(0, VL.SIM_POSTASTER)
        PostAster = import_module(Parameters.PostAsterFile)
        PostAsterSgl = getattr(PostAster, 'Single', None)

        VL.Logger("Running PostAster for '{}'\n".format(Parameters.Name),Print=True)
        os.makedirs(StudyDict['POSTASTER'],exist_ok=True)

        if not OutFile:
            err = PostAsterSgl(VL,StudyDict)
        else :
            with open(OutFile,'a') as f:
                with redirect_stdout(f), redirect_stderr(f):
                    err = PostAsterSgl(VL,StudyDict)
        if err:
            Error = ['PostAster',err]

    # Data structure which can be easily returned holdign additional information
    Returner = Namespace(Error=Error)
    # Add StudyDict to returned if it's changed
    if not OrigDict == StudyDict:
        Returner.StudyDict = StudyDict

    return Returner

def devRun(VL,**kwargs):
    kwargs.update(VL.GetArgParser()) # Update with any kwarg passed in the call

    ShowRes = kwargs.get('ShowRes', False)
    NumThreads = kwargs.get('NumThreads',1)

    # Run high throughput part in parallel
    NumSim = len(VL.SimData)
    NumThreads = min(NumThreads,NumSim)

    Arg0 = [VL]*NumSim
    Arg1 = list(VL.SimData.values())
    Arg2 = [kwargs]*NumSim
    if 1:
        pool = ProcessPool(nodes=NumThreads)
        Res = pool.map(PoolRun, Arg0, Arg1, Arg2)
    else :
        from pyina.launchers import MpiPool
        pool = MpiPool(nodes=NumThreads)
        Res = pool.map(PoolRun, Arg0, Arg1, Arg2)

    SimError = []
    for Name, Returner in zip(VL.SimData.keys(),Res):
        if Returner.Error:
            SimError.append(Name)
            VL.Logger("'{}' finished with errors".format(Name),Print=True)
        else :
            VL.Logger("'{}' completed successfully".format(Name), Print=True)
        if hasattr(Returner,'StudyDict'):
            VL.SimData[Name].update(Returner.StudyDict)

    if SimError:
        VL.Exit("The following Simulation routine(s) finished with errors:\n{}".format(SimError))

    PostAster = getattr(VL.Parameters_Master.Sim, 'PostAsterFile', None)
    if PostAster:
        sys.path.insert(0, VL.SIM_POSTASTER)
        PostAster = import_module(PostAster)
        if hasattr(PostAster, 'Combined'):
            VL.Logger('Combined function started', Print=True)
            if VL.mode in ('Interactive','Terminal'):
                err = PostAster.Combined(VL)
            else :
                with open(VL.LogFile, 'a') as f:
                    with redirect_stdout(f), redirect_stderr(f):
                        err = PostAster.Combined(VL)

            if err == None:
                VL.Logger('Combined function completed successfully', Print=True)
            else :
                VL.Exit("Combined function returned error '{}'".format(err))
