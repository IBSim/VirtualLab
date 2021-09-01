
import os
import sys
sys.dont_write_bytecode=True
from types import SimpleNamespace as Namespace
from importlib import import_module, reload
from contextlib import redirect_stderr, redirect_stdout
import shutil
import pickle

from Scripts.Common.VLPackages.Salome import Salome
from Scripts.Common.VLPackages.CodeAster import Aster
import Scripts.Common.VLFunctions as VLF
from Scripts.Common.VLParallel import VLPool

def CheckFile(Directory,fname,ext):
    if not fname:
        return True
    else:
        return os.path.isfile('{}/{}.{}'.format(Directory,fname,ext))

def Setup(VL,**kwargs):
    VL.SIM_SIM = "{}/Sim".format(VL.SIM_SCRIPTS)

    VL.SimData = {}
    SimDicts = VL.CreateParameters(VL.Parameters_Master, VL.Parameters_Var,'Sim')

    if not (kwargs.get('RunSim', True) and SimDicts): return

    os.makedirs(VL.STUDY_DIR, exist_ok=True)
    sys.path.insert(0,VL.SIM_SIM)
    for SimName, ParaDict in SimDicts.items():
        # Run checks
        # Check files exist
        if not CheckFile(VL.SIM_SIM,ParaDict.get('PreAsterFile'),'py'):
        	VL.Exit("PreAsterFile '{}.py' not in directory {}".format(ParaDict['PreAsterFile'],VL.SIM_SIM))
        if not CheckFile(VL.SIM_SIM,ParaDict.get('AsterFile'),'comm'):
        	VL.Exit("AsterFile '{}.comm' not in directory {}".format(ParaDict['AsterFile'],VL.SIM_SIM))
        if not CheckFile(VL.SIM_SIM, ParaDict.get('PostAsterFile'), 'py'):
        	VL.Exit("PostAsterFile '{}.py' not in directory {}".format(ParaDict['PostAsterFile'],VL.SIM_SIM))
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
        CALC_DIR = "{}/{}".format(VL.STUDY_DIR, SimName)
        StudyDict = {'Name':SimName,
                    'TMP_CALC_DIR':"{}/Sim/{}".format(VL.TEMP_DIR, SimName),
                    'CALC_DIR':CALC_DIR,
                    'PREASTER':"{}/PreAster".format(CALC_DIR),
                    'ASTER':"{}/Aster".format(CALC_DIR),
                    'POSTASTER':"{}/PostAster".format(CALC_DIR),
                    'MeshFile':"{}/{}.med".format(VL.MESH_DIR, ParaDict['Mesh']),
                    'Parameters':Namespace(**ParaDict)
                    }

        # Important information can be added to Data during any stage of the
        # simulation, and this will be saved to the location specified by the
        # value for the __file__ key
        StudyDict['Data'] = {'__file__':"{}/Data.pkl".format(StudyDict['CALC_DIR'])}
        StudyDict['LogFile'] = None
        if VL.mode in ('Headless','Continuous'):
            StudyDict['LogFile'] = "{}/Output.log".format(StudyDict['CALC_DIR'])
        elif VL.mode == 'Interactive':
            StudyDict['Interactive'] = True

        # Create tmp directory & add blank file to import in CodeAster
        # so we known the location of TMP_CALC_DIR
        os.makedirs(StudyDict['TMP_CALC_DIR'])
        with open("{}/IDDirVL.py".format(StudyDict['TMP_CALC_DIR']),'w') as f: pass

        # Add StudyDict to SimData dictionary
        VL.SimData[SimName] = StudyDict.copy()



def PoolRun(VL, StudyDict, kwargs):
    RunPreAster = kwargs.get('RunPreAster',True)
    RunAster = kwargs.get('RunAster', True)
    RunPostAster = kwargs.get('RunPostAster', True)

    Parameters = StudyDict["Parameters"]
    # Create CALC_DIR where results for this sim will be stored
    os.makedirs(StudyDict['CALC_DIR'],exist_ok=True)
    # Write Parameters used for this sim to CALC_DIR
    VLF.WriteData("{}/Parameters.py".format(StudyDict['CALC_DIR']), Parameters)

    if RunPreAster and hasattr(Parameters,'PreAsterFile'):
        PreAster = import_module(Parameters.PreAsterFile)
        PreAsterSgl = getattr(PreAster, 'Single',None)

        VL.Logger("Running PreAster for '{}'\n".format(Parameters.Name),Print=True)
        os.makedirs(StudyDict['PREASTER'],exist_ok=True)

        err = PreAsterSgl(VL,StudyDict)
        if err:
            return 'PreAster Error: {}'.format(err)

    if RunAster and hasattr(Parameters,'AsterFile'):
        VL.Logger("Running Aster for '{}'\n".format(Parameters.Name),Print=True)

        os.makedirs(StudyDict['ASTER'],exist_ok=True)
        # Create export file for CodeAster
        ExportFile = "{}/Export".format(StudyDict['ASTER'])
        CommFile = '{}/{}.comm'.format(VL.SIM_SIM, Parameters.AsterFile)
        MessFile = '{}/AsterLog'.format(StudyDict['ASTER'])
        Aster.ExportWriter(ExportFile, CommFile,
        							StudyDict["MeshFile"],
        							StudyDict['ASTER'],
                                    MessFile, **kwargs)

        # Create pickle of Dictionary
        pth = "{}/SimDict.pkl".format(StudyDict['TMP_CALC_DIR'])
        SimDict = {**StudyDict,'MATERIAL_DIR':VL.MATERIAL_DIR,'SIM_SCRIPTS':VL.SIM_SCRIPTS}
        with open(pth,'wb') as f:
        	pickle.dump(SimDict,f)

        # Run Simulation
        if 'Interactive' in StudyDict:
            SubProc = Aster.RunXterm(ExportFile, AddPath=[StudyDict['TMP_CALC_DIR']],
                                     tempdir=StudyDict['TMP_CALC_DIR'])
        else:
            SubProc = Aster.Run(ExportFile, AddPath=[StudyDict['TMP_CALC_DIR']])
        err = SubProc.wait()
        if err:
            return "Aster Error: Code {} returned".format(err)

        # Update if anything added to Dictionary
        with open(pth,'rb') as f:
            SimDictN = pickle.load(f)
            SimDictN.pop('MATERIAL_DIR');SimDictN.pop('SIM_SCRIPTS')
            if SimDictN != StudyDict:
                StudyDict.update(**SimDictN)

    if RunPostAster and hasattr(Parameters,'PostAsterFile'):
        PostAster = import_module(Parameters.PostAsterFile)
        PostAsterSgl = getattr(PostAster, 'Single', None)

        VL.Logger("Running PostAster for '{}'\n".format(Parameters.Name),Print=True)
        os.makedirs(StudyDict['POSTASTER'],exist_ok=True)

        err = PostAsterSgl(VL,StudyDict)
        if err:
             return 'PostAster Error: {}'.format(err)



def Run(VL,**kwargs):
    if not VL.SimData: return
    kwargs.update(VL.GetArgParser()) # Update with any kwarg passed in the call
    ShowRes = kwargs.get('ShowRes', False)
    NumThreads = kwargs.get('NumThreads',1)
    launcher = kwargs.get('launcher','Process')

    VL.Logger('\n### Starting Simulations ###\n', Print=True)

    # Run high throughput part in parallel
    NbSim = len(VL.SimData)
    SimDicts = list(VL.SimData.values())
    AddArgs = [[kwargs]*NbSim] #Additional arguments

    N = min(NumThreads,NbSim)

    Errorfnc = VLPool(VL,PoolRun,SimDicts,Args=AddArgs,launcher=launcher,N=N,onall=True)
    if Errorfnc:
        VL.Exit("The following Simulation routine(s) finished with errors:\n{}".format(Errorfnc))

    PostAster = getattr(VL.Parameters_Master.Sim, 'PostAsterFile', None)
    if PostAster and kwargs.get('RunPostAster', True):
        PostAster = import_module(PostAster)
        if hasattr(PostAster, 'Combined'):
            VL.Logger('Combined function started', Print=True)
            # sort this log part out to add to each log file
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

    VL.Logger('### Simulations Completed ###',Print=True)

    # Opens up all results in ParaVis
    if ShowRes:
    	print("\n### Opening results files in ParaVis ###\n")
    	ResFiles = {}
    	for SimName, StudyDict in VL.SimData.items():
    		for root, dirs, files in os.walk(StudyDict['CALC_DIR']):
    			for file in files:
    				fname, ext = os.path.splitext(file)
    				if ext == '.rmed':
    					ResFiles["{}_{}".format(SimName,fname)] = "{}/{}".format(root, file)
    	Script = "{}/ShowRes.py".format(Salome.Dir)
    	Salome.Run(Script, GUI=True, DataDict=ResFiles,tempdir=VL.TEMP_DIR)
