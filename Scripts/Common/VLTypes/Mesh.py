
import os
import sys
sys.dont_write_bytecode=True
from types import SimpleNamespace as Namespace
from importlib import import_module
import shutil
from Scripts.Common.VLPackages.Salome import Salome

import Scripts.Common.VLFunctions as VLF
from Scripts.Common.VLParallel import VLPool

def Setup(VL, **kwargs):
    VL.MESH_DIR = "{}/Meshes".format(VL.PROJECT_DIR)
    VL.SIM_MESH = "{}/Mesh".format(VL.SIM_SCRIPTS)

    VL.MeshData = {}
    MeshDicts = VL.CreateParameters(VL.Parameters_Master, VL.Parameters_Var,'Mesh')

    # if either MeshDicts is empty or RunMesh is False we will return
    if not (kwargs.get('RunMesh', True) and MeshDicts): return

    os.makedirs(VL.MESH_DIR, exist_ok=True)

    sys.path.insert(0, VL.SIM_MESH)

    for MeshName, ParaDict in MeshDicts.items():
        Parameters = Namespace(**ParaDict)
        # ====================================================================
        # Perform checks
        # Check that mesh file exists
        filepath = '{}/{}.py'.format(VL.SIM_MESH,Parameters.File)
        if not os.path.exists(filepath):
            VL.Exit(ErrorMessage('Mesh file\n{}\n does not exist'.format(filepath)))

        # Check Verify function, if it exists
        MeshFile = import_module(Parameters.File)
        if hasattr(MeshFile,'Verify'):
            error,warning = MeshFile.Verify(Parameters)
            if warning:
                mess = "Warning issed for mesh '{}':\n\n".format(MeshName)
                mess+= "\n\n".join(warning)
                print(VLF.WarningMessage(mess))

            if error:
                mess = "Error issued for mesh '{}':\n\n".format(MeshName)
                mess+= "\n\n".join(error)
                print(VLF.ErrorMessage(mess))
                VL.Exit()

        ## Checks complete ##

        MeshDict = {'Name':MeshName,
                    'MESH_FILE':"{}/{}.med".format(VL.MESH_DIR, MeshName),
                    'TMP_CALC_DIR':"{}/Mesh/{}".format(VL.TEMP_DIR, MeshName),
                    'Parameters':Parameters
                    }
        if VL.mode in ('Headless','Continuous'):
            MeshDict['LogFile'] = "{}/{}.log".format(VL.MESH_DIR,MeshName)
        else : MeshDict['LogFile'] = None

        os.makedirs(MeshDict['TMP_CALC_DIR'])

        VL.MeshData[MeshName] = MeshDict.copy()




def PoolRun(VL, MeshDict,**kwargs):
    # Write Parameters used to make the mesh to the mesh directory
    VLF.WriteData("{}/{}.py".format(VL.MESH_DIR, MeshDict['Name']), MeshDict['Parameters'])

    # Use a user-made MeshRun file if it exists. If not use the default one.
    if os.path.isfile('{}/MeshRun.py'.format(VL.SIM_MESH)):
        script = '{}/MeshRun.py'.format(VL.SIM_MESH)
    else:
        script = '{}/MeshRun.py'.format(Salome.Dir)

    err = Salome.Run(script, DataDict = MeshDict, AddPath=[VL.SIM_SCRIPTS,VL.SIM_MESH],
                     tempdir=MeshDict['TMP_CALC_DIR'])
    if err:
        return "Error in Salome run"

def Run(VL,**kwargs):
    if not VL.MeshData: return

    kwargs.update(VL.GetArgParser()) # Update with any kwarg passed in the call

    MeshCheck = kwargs.get('MeshCheck', None)
    ShowMesh = kwargs.get('ShowMesh', False)
    NumThreads = kwargs.get('NumThreads',1)
    launcher = kwargs.get('launcher','Process')

    '''
    MeshCheck routine which allows you to mesh in the GUI (useful for debugging).
    Currently only 1 mesh can be debugged at a time.
    VirtualLab will terminate once the GUI is closed.
    '''
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
        VL.Exit("Error: '{}' specified for MeshCheck is not one of meshes to be created.\n"\
                     "Meshes to be created are:{}".format(MeshCheck, list(VL.Data.keys())))

    VL.Logger('\n### Starting Meshing ###\n',Print=True)

    NbMeshes = len(VL.MeshData)
    MeshDicts = list(VL.MeshData.values())

    N = min(NumThreads,NbMeshes)

    Errorfnc = VLPool(VL,PoolRun,MeshDicts,launcher=launcher,N=N,onall=True)
    if Errorfnc:
        VL.Exit("\nThe following meshes finished with errors:\n{}".format(Errorfnc))

    VL.Logger('\n### Meshing Complete ###',Print=True)

    if ShowMesh:
        VL.Logger("\n### Opening mesh files in Salome ###\n",Print=True)
        ArgDict = {name:"{}/{}.med".format(VL.MESH_DIR, name) for name in VL.MeshData.keys()}
        Script = '{}/ShowMesh.py'.format(Salome.Dir)
        Salome.Run(Script, DataDict=ArgDict, tempdir=VL.TEMP_DIR, GUI=True)
        VL.Exit("\n### Terminating after mesh viewing ###")

def Cleanup():
    # TODO specify what we want to do at the end
    pass
