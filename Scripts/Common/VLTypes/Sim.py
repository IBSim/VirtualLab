
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

def Setup(VL,RunSim=True):
    VL.SIM_SIM = "{}/Sim".format(VL.SIM_SCRIPTS)

    VL.SimData = {}
    SimDicts = VL.CreateParameters(VL.Parameters_Master, VL.Parameters_Var,'Sim')

    if not (RunSim and SimDicts): return

    sys.path.insert(0,VL.SIM_SIM)
    for SimName, ParaDict in SimDicts.items():
        # Run checks
        # Check files exist
        if not CheckFile(VL.SIM_SIM,ParaDict.get('PreAsterFile'),'py'):
        	VL.Exit(VLF.ErrorMessage("PreAsterFile '{}.py' not in directory "\
                            "{}".format(ParaDict['PreAsterFile'],VL.SIM_SIM)))
        if not CheckFile(VL.SIM_SIM,ParaDict.get('AsterFile'),'comm'):
        	VL.Exit(VLF.ErrorMessage("AsterFile '{}.comm' not in directory "\
                            "{}".format(ParaDict['AsterFile'],VL.SIM_SIM)))
        if not CheckFile(VL.SIM_SIM, ParaDict.get('PostAsterFile'), 'py'):
        	VL.Exit(VLF.ErrorMessage("PostAsterFile '{}.py' not in directory "\
                            "{}".format(ParaDict['PostAsterFile'],VL.SIM_SIM)))
        # Check mesh will be available
        if not (ParaDict['Mesh'] in VL.MeshData or CheckFile(VL.MESH_DIR, ParaDict['Mesh'], 'med')):
        	VL.Exit(VLF.ErrorMessage("Mesh '{}' isn't being created and is "\
            "not in the mesh directory '{}'".format(ParaDict['Mesh'], VL.MESH_DIR)))
        # Check materials used
        Materials = ParaDict.get('Materials',[])
        if type(Materials)==str: Materials = [Materials]
        elif type(Materials)==dict: Materials = Materials.values()
        MatErr = [mat for mat in set(Materials) if not os.path.isdir('{}/{}'.format(VL.MATERIAL_DIR, mat))]
        if MatErr:
        		VL.Exit(VLF.ErrorMessage("Material(s) {} specified for {} not available.\n"\
        		"Please see the materials directory {} for options.".format(MatErr,SimName,VL.MATERIAL_DIR)))
        # Checks complete

        # Create dict of simulation specific information to be nested in SimData
        CALC_DIR = "{}/{}".format(VL.PROJECT_DIR, SimName)
        SimDict = {'Name':SimName,
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
        SimDict['Data'] = {'__file__':"{}/Data.pkl".format(SimDict['CALC_DIR'])}

        SimDict['LogFile'] = None
        if VL.mode in ('Headless','Continuous'):
            SimDict['LogFile'] = "{}/Output.log".format(SimDict['CALC_DIR'])
        elif VL.mode == 'Interactive':
            SimDict['Interactive'] = True

        # Create tmp directory & add blank file to import in CodeAster
        # so we known the location of TMP_CALC_DIR
        os.makedirs(SimDict['TMP_CALC_DIR'])
        with open("{}/IDDirVL.py".format(SimDict['TMP_CALC_DIR']),'w') as f: pass

        # Add SimDict to SimData dictionary
        VL.SimData[SimName] = SimDict.copy()



def PoolRun(VL, SimDict, Flags):
    RunPreAster,RunAster,RunPostAster = Flags

    Parameters = SimDict["Parameters"]
    # Create CALC_DIR where results for this sim will be stored
    os.makedirs(SimDict['CALC_DIR'],exist_ok=True)
    # Write Parameters used for this sim to CALC_DIR
    VLF.WriteData("{}/Parameters.py".format(SimDict['CALC_DIR']), Parameters)

    # ==========================================================================
    # Run pre aster step
    if RunPreAster and hasattr(Parameters,'PreAsterFile'):
        PreAster = import_module(Parameters.PreAsterFile)
        PreAsterSgl = getattr(PreAster, 'Single',None)

        VL.Logger("Running PreAster for '{}'\n".format(Parameters.Name),Print=True)
        os.makedirs(SimDict['PREASTER'],exist_ok=True)

        err = PreAsterSgl(VL,SimDict)
        if err:
            return 'PreAster Error: {}'.format(err)

    # ==========================================================================
    # Run aster step
    if RunAster and hasattr(Parameters,'AsterFile'):
        VL.Logger("Running Aster for '{}'\n".format(Parameters.Name),Print=True)

        os.makedirs(SimDict['ASTER'],exist_ok=True)

        #=======================================================================
        # Create export file for CodeAster
        ExportFile = "{}/Export".format(SimDict['ASTER'])
        CommFile = '{}/{}.comm'.format(VL.SIM_SIM, Parameters.AsterFile)
        MessFile = '{}/AsterLog'.format(SimDict['ASTER'])
        AsterSettings = getattr(Parameters,'AsterSettings',{})

        NbMpi = AsterSettings.get('mpi_nbcpu',1)
        if NbMpi >1:
            AsterSettings['actions'] = 'make_env'
            rep_trav =  "{}/CA".format(SimDict['TMP_CALC_DIR'])
            AsterSettings['rep_trav'] = rep_trav
            AsterSettings['version'] = 'stable_mpi'
            Aster.ExportWriter(ExportFile, CommFile, SimDict["MeshFile"],
            				   SimDict['ASTER'], MessFile, AsterSettings)
        else:
            Aster.ExportWriter(ExportFile, CommFile, SimDict["MeshFile"],
            				   SimDict['ASTER'], MessFile, AsterSettings)



        #=======================================================================
        # Write pickle of SimDict to file for code aster to find
        pth = "{}/SimDict.pkl".format(SimDict['TMP_CALC_DIR'])
        SimDictN = {**SimDict,'MATERIAL_DIR':VL.MATERIAL_DIR,'SIM_SCRIPTS':VL.SIM_SCRIPTS}
        with open(pth,'wb') as f:
        	pickle.dump(SimDictN,f)

        #=======================================================================
        # Run CodeAster
        if 'Interactive' in SimDict:
            # Run in x-term window
            err = Aster.RunXterm(ExportFile, AddPath=[SimDict['TMP_CALC_DIR']],
                                     tempdir=SimDict['TMP_CALC_DIR'])
        elif NbMpi>1:
            err = Aster.RunMPI(NbMpi, ExportFile, rep_trav, MessFile, SimDict['ASTER'], AddPath=[SimDict['TMP_CALC_DIR']])
        else:
            err = Aster.Run(ExportFile, AddPath=[SimDict['TMP_CALC_DIR']])

        if err:
            return "Aster Error: Code {} returned".format(err)

        #=======================================================================
        # Update SimDict with new information added during CodeAster run (if any)
        with open(pth,'rb') as f:
            SimDictN = pickle.load(f)
            SimDictN.pop('MATERIAL_DIR');SimDictN.pop('SIM_SCRIPTS')
            if SimDictN != SimDict:
                SimDict.update(**SimDictN)

    # ==========================================================================
    # Run post aster step
    if RunPostAster and hasattr(Parameters,'PostAsterFile'):
        PostAster = import_module(Parameters.PostAsterFile)
        PostAsterSgl = getattr(PostAster, 'Single', None)

        VL.Logger("Running PostAster for '{}'\n".format(Parameters.Name),Print=True)
        os.makedirs(SimDict['POSTASTER'],exist_ok=True)

        err = PostAsterSgl(VL,SimDict)
        if err:
             return 'PostAster Error: {}'.format(err)


def Run(VL, RunPreAster=True, RunAster=True, RunPostAster=True, ShowRes=False):
    if not VL.SimData: return

    # ==========================================================================
    # Run Sim routine

    VL.Logger('\n### Starting Simulations ###\n', Print=True)

    # Run high throughput part in parallel
    NbSim = len(VL.SimData)
    SimDicts = list(VL.SimData.values())
    Flags = [RunPreAster,RunAster,RunPostAster]
    AddArgs = [[Flags]*NbSim] #Additional arguments
    N = min(VL._NbThreads,NbSim)

    Errorfnc = VLPool(VL,PoolRun,SimDicts,Args=AddArgs,launcher=VL._Launcher,N=N,onall=True)
    if Errorfnc:
        VL.Exit(VLF.ErrorMessage("The following Simulation routine(s) finished with errors:\n{}".format(Errorfnc)))

    VL.Logger('### Simulations Completed ###',Print=True)

    # ==========================================================================
    # Open up all results in ParaVis

    if ShowRes:
    	print("\n### Opening results files in ParaVis ###\n")
    	ResFiles = {}
    	for SimName, SimDict in VL.SimData.items():
    		for root, dirs, files in os.walk(SimDict['CALC_DIR']):
    			for file in files:
    				fname, ext = os.path.splitext(file)
    				if ext == '.rmed':
    					ResFiles["{}_{}".format(SimName,fname)] = "{}/{}".format(root, file)
    	Script = "{}/ShowRes.py".format(Salome.Dir)
    	Salome.Run(Script, GUI=True, DataDict=ResFiles,tempdir=VL.TEMP_DIR)
