
import os
import sys
sys.dont_write_bytecode=True
from types import SimpleNamespace as Namespace
from importlib import import_module
# from multiprocessing import Process, Pool
from pathos.multiprocessing import ProcessPool
from ..VLPackages import Salome

def Setup(VL, **kwargs):
    VL.MESH_DIR = "{}/Meshes".format(VL.PROJECT_DIR)
    os.makedirs(VL.MESH_DIR, exist_ok=True)

    VL.MeshData = {}

    if not kwargs.get('RunMesh', True): return

    MeshDicts = VL.CreateParameters(VL.Parameters_Master, VL.Parameters_Var,'Mesh')

    VL.GEOM_DIR = '{}/Geom'.format(VL.TMP_DIR)
    os.makedirs(VL.GEOM_DIR, exist_ok=True)

    sys.path.insert(0, VL.SIM_MESH)

    for MeshName, ParaDict in MeshDicts.items():
    	## Run checks ##
    	# Check that mesh file exists
    	if not os.path.exists('{}/{}.py'.format(VL.SIM_MESH,ParaDict['File'])):
    		VL.Exit("Mesh file '{}' does not exist in {}".format(ParaDict['File'], VL.SIM_MESH))

    	MeshFile = import_module(ParaDict['File'])
    	try :
    		err = MeshFile.GeomError(Namespace(**ParaDict))
    		if err: VL.Exit("GeomError in '{}' - {}".format(MeshDict['Name'], err))
    	except AttributeError:
    		pass
    	## Checks complete ##

    	VL.WriteModule("{}/{}.py".format(VL.GEOM_DIR, MeshName), ParaDict)
    	VL.MeshData[MeshName] = Namespace(**ParaDict)


def PoolRun(VL,MeshName,**kwargs):

    script = '{}/VLPackages/Salome/MeshRun.py'.format(VL.COM_SCRIPTS)
    AddPath = [VL.SIM_MESH, VL.GEOM_DIR]
    Returnfile = "{}/{}_RC.txt".format(VL.GEOM_DIR,MeshName) # File where salome can write an exit status to
    ArgDict = {'Name':MeshName,
    				'MESH_FILE':"{}/{}.med".format(VL.MESH_DIR, MeshName),
    				'RCfile':Returnfile}
    if os.path.isfile('{}/config.py'.format(VL.SIM_MESH)): ArgDict["ConfigFile"] = True

    err = VL.Salome.Run(script, AddPath=AddPath, ArgDict=ArgDict)
    if err:
        VL.Logger("Error code {} returned in Salome run".format(err))
        return err

    if os.path.isfile(Returnfile):
        with open(Returnfile,'r') as f:
            return int(f.readline())

def devRun(VL,**kwargs):
    kwargs.update(VL.GetArgParser()) # Update with any kwarg passed in the call
    
    MeshCheck = kwargs.get('MeshCheck', None)
    ShowMesh = kwargs.get('ShowMesh', False)
    NumThreads = kwargs.get('NumThreads',1)

    # MeshCheck routine which allows you to mesh in the GUI (Used for debugging).
    # The script will terminate after this routine
    if MeshCheck and MeshCheck in VL.Data.keys():
        VL.Logger('### Meshing {} in GUI ###\n'.format(MeshCheck), Print=True)
        # The file MeshParaFile is passed to MeshScript to create the mesh in the GUI
        MeshParaFile = "{}/{}.py".format(VL.GEOM_DIR,MeshCheck)
        MeshScript = "{}/{}.py".format(VL.SIM_MESH, VL.Data[MeshCheck].File)

        VL.Salome.Run(MeshScript, ArgList=[MeshParaFile], GUI=True)
        VL.Exit('Terminating after checking mesh')

    elif MeshCheck and MeshCheck not in VL.Data.keys():
        VL.Exit("Error: '{}' specified for MeshCheck is not one of meshes to be created.\n"\
        	         "Meshes to be created are:{}".format(MeshCheck, list(VL.Data.keys())))

    VL.Logger('\n### Starting Meshing ###\n',Print=True)

    NumMeshes = len(VL.MeshData)
    NumThreads = min(NumThreads,NumMeshes)

    Arg0 = [VL]*len(VL.MeshData)
    Arg1 = list(VL.MeshData.keys())

    if 1:
        pool = ProcessPool(nodes=NumThreads)
        Res = pool.map(PoolRun, Arg0, Arg1)
    else :
        from pyina.launchers import MpiPool
        pool = MpiPool(nodes=NumThreads)
        Res = pool.map(PoolRun, Arg0, Arg1)

    MeshError = []
    for Name, RC in zip(Arg1,Res):
        if RC:
            MeshError.append(Name)
            VL.Logger("'{}' finished with errors".format(Name),Print=True)
        else :
            VL.Logger("'{}' completed successfully".format(Name), Print=True)

    if MeshError:
        VL.Exit("The following Meshes finished with errors:\n{}".format(MeshError),KeepDirs=['Geom'])

    VL.Logger('\n### Meshing Completed ###',Print=True)
    if ShowMesh:
        VL.Logger("Opening mesh files in Salome",Print=True)
        ArgDict = {name:"{}/{}.med".format(VL.MESH_DIR, name) for name in VL.MeshData.keys()}
        Script = '{}/VLPackages/Salome/ShowMesh.py'.format(VL.COM_SCRIPTS)
        VL.Salome.Run(Script, ArgDict=ArgDict, GUI=True)
        VL.Exit("Terminating after mesh viewing")

def Cleanup():
    # TODO specify what we want to do at the end
    pass
