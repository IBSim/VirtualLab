
import os
import sys
sys.dont_write_bytecode=True
from types import SimpleNamespace as Namespace
from importlib import import_module
import shutil
from Scripts.Common.VLPackages import SalomeRun

import Scripts.Common.VLFunctions as VLF

def Setup(VL, **kwargs):
    VL.MESH_DIR = "{}/Meshes".format(VL.PROJECT_DIR)

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
        filepath = '{}/{}.py'.format(VL.SIM_MESH,ParaDict['File'])
        if not os.path.exists(filepath):
            VL.Exit(ErrorMessage('Mesh file\n{}\n does not exist'.format(filepath)))

        # Check Verify function, if it exists
        MeshFile = import_module(ParaDict['File'])
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

        Mdict = {'Name':MeshName,
                 'MESH_FILE':"{}/{}.med".format(VL.MESH_DIR, MeshName),
                 'Parameters':Parameters}
        if VL.mode in ('Headless','Continuous'):
            Mdict['LogFile'] = "{}/{}.log".format(VL.MESH_DIR,MeshName)
        else : Mdict['LogFile'] = None
        VL.MeshData[MeshName] = Mdict.copy()


def PoolRun(VL, MeshDict,**kwargs):
    # Write Parameters used to make the mesh to the mesh directory
    VLF.WriteData("{}/{}.py".format(VL.MESH_DIR, MeshDict['Name']), MeshDict['Parameters'])

    if os.path.isfile('{}/MeshRun.py'.format(VL.SIM_MESH)):
        script = '{}/MeshRun.py'.format(VL.SIM_MESH)
    else:
        script = '{}/VLPackages/Salome/MeshRun.py'.format(VL.COM_SCRIPTS)

    err = SalomeRun(script, DataDict = MeshDict, AddPath=[VL.SIM_SCRIPTS,VL.SIM_MESH])
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
        VL.Logger('\n### Meshing {} in GUI ###\n'.format(MeshCheck), Print=True)

        if os.path.isfile('{}/MeshRun.py'.format(VL.SIM_MESH)):
            script = '{}/MeshRun.py'.format(VL.SIM_MESH)
        else:
            script = '{}/VLPackages/Salome/MeshRun.py'.format(VL.COM_SCRIPTS)
        VL.MeshData[MeshCheck]['Debug'] = True
        SalomeRun(script, DataDict = VL.MeshData[MeshCheck],
                        AddPath=[VL.SIM_MESH,VL.SIM_SCRIPTS], GUI=True)

        VL.Exit('Terminating after checking mesh')

    elif MeshCheck and MeshCheck not in VL.MeshData.keys():
        VL.Exit("Error: '{}' specified for MeshCheck is not one of meshes to be created.\n"\
                     "Meshes to be created are:{}".format(MeshCheck, list(VL.Data.keys())))

    VL.Logger('\n### Starting Meshing ###\n',Print=True)

    NbMeshes = len(VL.MeshData)
    MeshDicts = list(VL.MeshData.values())
    PoolArgs = [[VL]*NbMeshes, MeshDicts]

    N = min(NumThreads,NbMeshes)

    if launcher == 'Sequential':
        Res = []
        for args in zip(*PoolArgs):
            ret = VLF.VLPool(PoolRun,*args)
            Res.append(ret)
    elif launcher == 'Process':
        from pathos.multiprocessing import ProcessPool
        pool = ProcessPool(nodes=N, workdir=VL.TEMP_DIR)
        Res = pool.map(VLF.VLPool,[PoolRun]*NbMeshes, *PoolArgs)
    elif launcher == 'MPI':
        from pyina.launchers import MpiPool
        # Ensure that all paths added to sys.path are visible pyinas MPI subprocess
        addpath = set(sys.path) - set(VL._pypath) # group subtraction
        addpath = ":".join(addpath) # write in unix style
        PyPath_orig = os.environ.get('PYTHONPATH',"")
        os.environ["PYTHONPATH"] = "{}:{}".format(addpath,PyPath_orig)

        onall = kwargs.get('onall',True) # Do we want 1 mpi worked to delegate and not compute (False if so)
        if not onall and NumThreads > N: N=N+1 # Add 1 if extra threads available for 'delegator'

        pool = MpiPool(nodes=N,source=True, workdir=VL.TEMP_DIR)
        Res = pool.map(VLF.VLPool,[PoolRun]*NbMeshes, *PoolArgs, onall=onall)

        # reset environment back to original
        os.environ["PYTHONPATH"] = PyPath_orig

    Errorfnc = VLF.VLPoolReturn(MeshDicts,Res)
    if Errorfnc:
        VL.Exit("\nThe following meshes finished with errors:\n{}".format(Errorfnc),KeepDirs=['Geom'])

    VL.Logger('\n### Meshing Complete ###',Print=True)

    if ShowMesh:
        VL.Logger("\n### Opening mesh files in Salome ###\n",Print=True)
        ArgDict = {name:"{}/{}.med".format(VL.MESH_DIR, name) for name in VL.MeshData.keys()}
        Script = '{}/VLPackages/Salome/ShowMesh.py'.format(VL.COM_SCRIPTS)
        SalomeRun(Script, DataDict=ArgDict, GUI=True)
        VL.Exit("\n### Terminating after mesh viewing ###")

def Cleanup():
    # TODO specify what we want to do at the end
    pass
