import os
import sys
sys.dont_write_bytecode=True
from types import SimpleNamespace as Namespace

from Scripts.Common.VLPackages.Salome import Salome
import Scripts.Common.VLFunctions as VLF
from Scripts.Common.VLParallel import VLPool

def Setup(VL, RunMesh=True, Import=False):
    VL.MESH_DIR = "{}/Meshes".format(VL.PROJECT_DIR)
    VL.SIM_MESH = "{}/Mesh".format(VL.SIM_SCRIPTS)

    VL.MeshData = {}
    MeshDicts = VL.CreateParameters(VL.Parameters_Master, VL.Parameters_Var,'Mesh')

    # if either MeshDicts is empty or RunMesh is False we will return
    if not (RunMesh and MeshDicts): return
    sys.path.insert(0, VL.SIM_MESH)

    FileDict = {} # Something we want to keep track of
    for MeshName, ParaDict in MeshDicts.items():
        MeshPath = "{}/{}".format(VL.MESH_DIR, MeshName)
        if Import:
            VLF.ImportUpdate("{}.py".format(MeshPath), ParaDict)

        Parameters = Namespace(**ParaDict)

        # ======================================================================
        # get file path & perform checks
        # default name is Single
        file_name,func_name = VLF.FileFuncSplit(ParaDict['File'],'Create')

        if (file_name,func_name) not in FileDict:
            # Check file in directory & get path
            FilePath = VL.GetFilePath([VL.SIM_MESH], file_name,
                                      file_ext='py', exit_on_error=True)
            # Check function func_name is in the file
            a = VL.GetFunction(FilePath,func_name,exit_on_error=True)
            File_func = [FilePath,func_name]
            FileDict[(file_name,func_name)] = File_func
        else:
            File_func = FileDict[(file_name,func_name)]

        # ======================================================================
        # Verify mesh parameters
        Verify = VLF.GetFunc(FilePath,'Verify')
        if Verify != None:
            error, warning = Verify(Parameters)
            if warning:
                print(VLF.WarningMessage('Issue with mesh {}.'\
                        '\n\n{}'.format(MeshName, warning)))
            if error:
                VL.Exit(VLF.ErrorMessage('Issue with mesh {}.'\
                        '\n\n{}'.format(MeshName, error)))

        # ======================================================================
        # Create dictionary for each analysis
        MeshDict = {'Name':MeshName,
                    'MESH_FILE':"{}.med".format(MeshPath),
                    'TMP_CALC_DIR':"{}/Mesh/{}".format(VL.TEMP_DIR, MeshName),
                    'FileInfo':File_func,
                    'Parameters':Parameters
                    }
        if VL.mode in ('Headless','Continuous'):
            MeshDict['LogFile'] = "{}/{}.log".format(VL.MESH_DIR,MeshName)
        else : MeshDict['LogFile'] = None

        os.makedirs(MeshDict['TMP_CALC_DIR'])

        VL.MeshData[MeshName] = MeshDict.copy()

def PoolRun(VL, MeshDict,**kwargs):
    # Create directory for meshes.
    # This method supports meshes nested in sub-directories
    Meshfname = os.path.splitext(MeshDict['MESH_FILE'])[0]
    os.makedirs(os.path.dirname(Meshfname),exist_ok=True)

    # Write Parameters used to make the mesh to the mesh directory
    VLF.WriteData("{}.py".format(Meshfname), MeshDict['Parameters'])

    # Use a user-made MeshRun file if it exists. If not use the default one.
    if os.path.isfile('{}/MeshRun.py'.format(VL.SIM_MESH)):
        script = '{}/MeshRun.py'.format(VL.SIM_MESH)
    else:
        script = '{}/MeshRun.py'.format(Salome.Dir)

    err = Salome.Run(script, DataDict = MeshDict, AddPath=[VL.SIM_SCRIPTS,VL.SIM_MESH],
                     tempdir=MeshDict['TMP_CALC_DIR'])
    if err:
        return "Error in Salome run"

def Run(VL,MeshCheck=None,ShowMesh=False):
    if not VL.MeshData: return

    #===========================================================================
    # MeshCheck allows you to mesh in the GUI (for debugging).Currently only 1
    # mesh can be debugged at a time. VirtualLab terminates when GUI is closed.

    if MeshCheck and MeshCheck in VL.MeshData.keys():
        MeshDict = VL.MeshData[MeshCheck]
        VL.Logger('\n### Meshing {} in GUI ###\n'.format(MeshCheck), Print=True)

        if os.path.isfile('{}/MeshRun.py'.format(VL.SIM_MESH)):
            script = '{}/MeshRun.py'.format(VL.SIM_MESH)
        else:
            script = '{}/MeshRun.py'.format(Salome.Dir)
        MeshDict['Debug'] = True
        Salome.Run(script, DataDict = MeshDict,tempdir=MeshDict['TMP_CALC_DIR'],
                   AddPath=[VL.SIM_MESH,VL.SIM_SCRIPTS], GUI=True)

        VL.Exit('Terminating after checking mesh')

    elif MeshCheck and MeshCheck not in VL.MeshData.keys():
        VL.Exit(VLF.ErrorMessage("'{}' specified for MeshCheck is not one of meshes to be created.\n"\
                     "Meshes to be created are:{}".format(MeshCheck, list(VL.Data.keys()))))

    # ==========================================================================
    # Run Mesh routine

    VL.Logger('~~~~~~~~~~~~~~~~~~~~~~~~\n'\
              '~~~ Starting Meshing ~~~\n'\
              '~~~~~~~~~~~~~~~~~~~~~~~~\n',Print=True)

    NbMeshes = len(VL.MeshData)
    MeshDicts = list(VL.MeshData.values())

    Errorfnc = VLPool(VL,PoolRun,MeshDicts)
    if Errorfnc:
        VL.Exit(VLF.ErrorMessage("\nThe following meshes finished with errors:\n{}".format(Errorfnc)),
                Cleanup=False)

    VL.Logger('~~~~~~~~~~~~~~~~~~~~~~~~\n'\
              '~~~ Meshing Complete ~~~\n'\
              '~~~~~~~~~~~~~~~~~~~~~~~~\n',Print=True)

    # ==========================================================================
    # Open meshes in GUI to view

    if ShowMesh:
        VL.Logger("\n### Opening mesh files in Salome ###\n",Print=True)
        ArgDict = {name:"{}/{}.med".format(VL.MESH_DIR, name) for name in VL.MeshData.keys()}
        Script = '{}/ShowMesh.py'.format(Salome.Dir)
        Salome.Run(Script, DataDict=ArgDict, tempdir=VL.TEMP_DIR, GUI=True)
        VL.Exit("\n### Terminating after mesh viewing ###")

def Cleanup():
    # TODO specify what we want to do at the end
    pass
