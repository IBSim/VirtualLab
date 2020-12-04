
import os
import sys
sys.dont_write_bytecode=True
from types import SimpleNamespace as Namespace
from importlib import import_module
# from multiprocessing import Process, Pool
from pathos.multiprocessing import ProcessPool
from ..VLPackages import Salome

class Mesh():
    def __init__(self,VL,**kwargs):

        # Take what's necessary from VL class
        # Variables
        self.TMP_DIR = VL.TMP_DIR
        self.LogFile = VL.LogFile
        self.SIM_MESH = VL.SIM_MESH
        self.MESH_DIR = VL.MESH_DIR
        self.COM_SCRIPTS = VL.COM_SCRIPTS
        self.GEOM_DIR = '{}/Geom'.format(self.TMP_DIR)
        self.Logger = VL.Logger
        self.Exit = VL.Exit
        # Functions
        self.Data = {}
        self.Salome = Salome.Salome(self)

        self.Setup(VL)

    def Setup(self, VL,**kwargs):
        os.makedirs(self.MESH_DIR, exist_ok=True)
        os.makedirs(self.GEOM_DIR, exist_ok=True)
        sys.path.insert(0, self.SIM_MESH)

        # Create dictionaries from Mesh attribute of Parameters Master & Var
        MeshDicts = VL.CreateParameters(VL.Parameters_Master, VL.Parameters_Var,'Mesh')

        for MeshName, ParaDict in MeshDicts.items():
        	## Run checks ##
        	# Check that mesh file exists
        	if not os.path.exists('{}/{}.py'.format(self.SIM_MESH,ParaDict['File'])):
        		self.Exit("Mesh file '{}' does not exist in {}".format(ParaDict['File'], self.SIM_MESH))

        	MeshFile = import_module(ParaDict['File'])
        	try :
        		err = MeshFile.GeomError(Namespace(**ParaDict))
        		if err: self.Exit("GeomError in '{}' - {}".format(MeshDict['Name'], err))
        	except AttributeError:
        		pass
        	## Checks complete ##

        	VL.WriteModule("{}/{}.py".format(self.GEOM_DIR, MeshName), ParaDict)
        	self.Data[MeshName] = Namespace(**ParaDict)


    def PoolRun(self,MeshName,**kwargs):
    	script = '{}/VLPackages/Salome/MeshRun.py'.format(self.COM_SCRIPTS)
    	AddPath = [self.SIM_MESH, self.GEOM_DIR]
    	ArgDict = {'Name':MeshName,
    					'MESH_FILE':"{}/{}.med".format(self.MESH_DIR, MeshName),
    					'RCfile':"{}/{}_RC.txt".format(self.GEOM_DIR,MeshName)}
    	if os.path.isfile('{}/config.py'.format(self.SIM_MESH)): ArgDict["ConfigFile"] = True

    	self.Salome.TestRun(script, AddPath=AddPath, ArgDict=ArgDict)

    def Run(self,**kwargs):
        MeshCheck = kwargs.get('MeshCheck', None)
        ShowMesh = kwargs.get('ShowMesh', False)
        NumThreads = kwargs.get('NumThreads',1)

        # MeshCheck routine which allows you to mesh in the GUI (Used for debugging).
        # The script will terminate after this routine
        if MeshCheck and MeshCheck in self.Data.keys():
            self.Logger('### Meshing {} in GUI ###\n'.format(MeshCheck), Print=True)
            # The file MeshParaFile is passed to MeshScript to create the mesh in the GUI
            MeshParaFile = "{}/{}.py".format(self.GEOM_DIR,MeshCheck)
            MeshScript = "{}/{}.py".format(self.SIM_MESH, self.Data[MeshCheck].File)

            SubProc = self.Salome.Run(MeshScript, ArgList=[MeshParaFile], GUI=True)
            SubProc.wait()
            self.Exit('Terminating after checking mesh')

        elif MeshCheck and MeshCheck not in self.Data.keys():
            self.Exit("Error: '{}' specified for MeshCheck is not one of meshes to be created.\n"\
            	         "Meshes to be created are:{}".format(MeshCheck, list(self.Data.keys())))

        self.Logger('\n### Starting Meshing ###\n',Print=True)

        NumMeshes = len(self.Data)
        NumThreads = min(NumThreads,NumMeshes)

        Arg1 = list(self.Data.keys())

        if True:
            pool = ProcessPool(nodes=NumThreads)
            pool.map(self.PoolRun, Arg1)
        else :
            Arg0 = [self]*len(self.Data)
            from pyina.ez_map import ez_map
            from pyina.launchers import  mpirun_launcher, srun_launcher
            ez_map(extRun, Arg0, Arg1, nodes=NumThreads,launcher=mpirun_launcher)


        self.Logger('\n### Meshing Completed ###',Print=True)
        if ShowMesh:
            self.Logger("Opening mesh files in Salome",Print=True)
            ArgDict = {name:"{}/{}.med".format(self.MESH_DIR, name) for name in self.Data.keys()}
            Script = '{}/VLPackages/Salome/ShowMesh.py'.format(self.COM_SCRIPTS)
            SubProc = self.Salome.Run(Script, ArgDict=ArgDict, GUI=True)
            SubProc.wait()
            self.Exit("Terminating after mesh viewing")

    def Cleanup():
        # TODO specify what we want to do at the end
        pass

def extRun(Meta,arg1):
    Mesh.PoolRun(Meta,arg1)
