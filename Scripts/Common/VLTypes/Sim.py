
import os
import sys
sys.dont_write_bytecode=True
from types import SimpleNamespace as Namespace
import pickle

from Scripts.Common.VLPackages.Salome import Salome
from Scripts.Common.VLPackages.CodeAster import Aster
import Scripts.Common.VLFunctions as VLF
from Scripts.Common.VLParallel import VLPool

def Setup(VL,RunSim=True,Import=False):
    VL.SIM_SIM = "{}/Sim".format(VL.SIM_SCRIPTS)

    VL.SimData = {}
    SimDicts = VL.CreateParameters(VL.Parameters_Master, VL.Parameters_Var,'Sim')

    if not (RunSim and SimDicts): return

    sys.path.insert(0,VL.SIM_SIM)

    AsterFiles,PFiles,AsterFileInfo,PFileInfo = [],[],[],[]
    AsterFileDict, PFileDict = {}, {}
    Meshes,Materials = [],[]
    for SimName, ParaDict in SimDicts.items():
        CALC_DIR = "{}/{}".format(VL.PROJECT_DIR, SimName)
        if Import:
            ParaDict = VLF.ImportUpdate("{}/Parameters.py".format(CALC_DIR), ParaDict)

        # ======================================================================
        # Create dictionary for each analysis
        SimDict = {'Name':SimName,
                    'TMP_CALC_DIR':"{}/Sim/{}".format(VL.TEMP_DIR, SimName),
                    'CALC_DIR':CALC_DIR,
                    'PREASTER':"{}/PreAster".format(CALC_DIR),
                    'ASTER':"{}/Aster".format(CALC_DIR),
                    'POSTASTER':"{}/PostAster".format(CALC_DIR),
                    'MeshFile':"{}/{}.med".format(VL.MESH_DIR, ParaDict['Mesh']),
                    'Parameters':Namespace(**ParaDict)
                    }

        # ======================================================================
        # get file path & perform checks
        AsterFile = ParaDict.get('AsterFile')
        if AsterFile:
            if AsterFile not in AsterFileDict:
                # Check file in directory & get path
                AsterFilePath = VL.GetFilePath([VL.SIM_SIM], AsterFile,
                                          file_ext='comm', exit_on_error=True)
                AsterFileDict[AsterFile] = AsterFilePath
            else:
                AsterFilePath = AsterFileDict[AsterFile]
            SimDict['AsterFile'] = AsterFilePath

        PreFile,PostFile = ParaDict.get('PreAsterFile'),ParaDict.get('PostAsterFile')
        if PreFile or PostFile:
            for fname, name in zip([PreFile,PostFile],['PreFile','PostFile']):
                if fname is None: continue
                # default name is Single
                file_name,func_name = VLF.FileFuncSplit(fname,'Single')
                if (file_name,func_name) not in PFileDict:
                    FilePath = VL.GetFilePath([VL.SIM_SIM,VL.VLRoutine_SCRIPTS], file_name,
                                              file_ext='py', exit_on_error=True)
                    # Check function func_name is in the file
                    VL.GetFunction(FilePath,func_name,exit_on_error=True)
                    File_func = [FilePath,func_name]
                    PFileDict[(file_name,func_name)] = File_func
                else:
                    File_func = PFileDict[(file_name,func_name)]
                SimDict[name] = File_func

        # ======================================================================
        # Check meshes
        Mesh = ParaDict.get('Mesh')
        if Mesh and Mesh not in Meshes: Meshes.append(Mesh)
        # Check materials
        Mat = ParaDict.get('Materials',[])
        if type(Mat)==str: Materials.append(Mat)
        elif type(Mat)==dict: Materials.extend(list(Mat.values()))

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

    # Check mesh exists
    for MeshName in Meshes:
        MeshCreate = MeshName in VL.MeshData
        if not MeshCreate:
            FilePath = VL.GetFilePath(VL.MESH_DIR, MeshName,file_ext='med',exit_on_error=False)
            if FilePath is None:
                VL.Exit(VLF.ErrorMessage("Mesh '{}' isn't being created and is "\
                "not in the mesh directory {}".format(MeshName, VL.MESH_DIR)))
    # Check material
    for Material in set(Materials):
        MatExist = os.path.isdir("{}/{}".format(VL.MATERIAL_DIR,Material))
        if not MatExist:
            VL.Exit(VLF.ErrorMessage("Material '{}' not available.\n"\
            "Please see the materials directory {} for options.".format(Material,VL.MATERIAL_DIR)))

    # ==========================================================================

def PoolRun(VL, SimDict, Flags):
    RunPreAster,RunAster,RunPostAster = Flags

    Parameters = SimDict["Parameters"]
    # Create CALC_DIR where results for this sim will be stored
    os.makedirs(SimDict['CALC_DIR'],exist_ok=True)
    # Write Parameters used for this sim to CALC_DIR
    VLF.WriteData("{}/Parameters.py".format(SimDict['CALC_DIR']), Parameters)

    # ==========================================================================
    # Run pre aster step
    if RunPreAster and 'PreFile' in SimDict:
        VL.Logger("Running PreAster for '{}'\n".format(Parameters.Name),Print=True)
        os.makedirs(SimDict['PREASTER'],exist_ok=True)

        PreAsterFnc = VLF.GetFunc(*SimDict['PreFile'])
        err = PreAsterFnc(VL,SimDict)
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
        CommFile = SimDict['AsterFile']
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
    if RunPostAster and 'PostFile' in SimDict:
        VL.Logger("Running PostAster for '{}'\n".format(Parameters.Name),Print=True)
        os.makedirs(SimDict['POSTASTER'],exist_ok=True)

        PostAsterFnc = VLF.GetFunc(*SimDict['PostFile'])

        err = PostAsterFnc(VL,SimDict)
        if err:
             return 'PostAster Error: {}'.format(err)


def Run(VL, RunPreAster=True, RunAster=True, RunPostAster=True, ShowRes=False):
    if not VL.SimData: return

    # ==========================================================================
    # Run Sim routine

    VL.Logger('~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n'\
              '~~~ Starting Simulations ~~~\n'\
              '~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n', Print=True)

    # Run high throughput part in parallel
    NbSim = len(VL.SimData)
    SimDicts = list(VL.SimData.values())
    Flags = [RunPreAster,RunAster,RunPostAster]
    AddArgs = [[Flags]*NbSim] #Additional arguments

    Errorfnc = VLPool(VL,PoolRun,SimDicts,Args=AddArgs)
    if Errorfnc:
        VL.Exit(VLF.ErrorMessage("The following Simulation routine(s) finished with errors:\n{}".format(Errorfnc)),
                Cleanup=False)

    VL.Logger('~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n'\
              '~~~ Simulations Complete ~~~\n'\
              '~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n',Print=True)

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
