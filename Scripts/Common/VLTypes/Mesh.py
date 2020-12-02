
import os
import sys
sys.dont_write_bytecode=True
from types import SimpleNamespace as Namespace
from importlib import import_module
# from multiprocessing import Process, Pool
from pathos.multiprocessing import ProcessPool

class Mesh():
    def __init__(self,VL):
        self.VL = VL
        self.Data = {}

    def Setup(self,**kwargs):
        self.GEOM_DIR = '{}/Geom'.format(self.VL.TMP_DIR)
        os.makedirs(self.VL.MESH_DIR, exist_ok=True)
        os.makedirs(self.GEOM_DIR, exist_ok=True)
        # Get dictionary of mesh parameters using Parameters_Master and Parameters_Var
        MeshDicts = self.VL.CreateParameters(self.VL.Parameters_Master,self.VL.Parameters_Var,'Mesh')
        sys.path.insert(0, self.VL.SIM_MESH)

        for MeshName, ParaDict in MeshDicts.items():
        	## Run checks ##
        	# Check that mesh file exists
        	if not os.path.exists('{}/{}.py'.format(self.VL.SIM_MESH,ParaDict['File'])):
        		self.VL.Exit("Mesh file '{}' does not exist in {}".format(ParaDict['File'], self.VL.SIM_MESH))

        	MeshFile = import_module(ParaDict['File'])
        	try :
        		err = MeshFile.GeomError(Namespace(**ParaDict))
        		if err: self.VL.Exit("GeomError in '{}' - {}".format(MeshDict['Name'], err))
        	except AttributeError:
        		pass
        	## Checks complete ##

        	self.VL.WriteModule("{}/{}.py".format(self.GEOM_DIR, MeshName), ParaDict)
        	self.Data[MeshName] = Namespace(**ParaDict)

    def __Run__(self,MeshName,**kwargs):
    	script = '{}/VLPackages/Salome/MeshRun.py'.format(self.VL.COM_SCRIPTS)
    	AddPath = [self.VL.SIM_MESH, self.GEOM_DIR]
    	ArgDict = {'Name':MeshName,
    					'MESH_FILE':"{}/{}.med".format(self.VL.MESH_DIR, MeshName),
    					'RCfile':"{}/{}_RC.txt".format(self.GEOM_DIR,MeshName)}
    	if os.path.isfile('{}/config.py'.format(self.VL.SIM_MESH)): ArgDict["ConfigFile"] = True
    	# print(ArgDict)
    	self.VL.Salome.TestRun(script, AddPath=AddPath, ArgDict=ArgDict)

    def Run(self,**kwargs):
        MeshCheck = kwargs.get('MeshCheck', None)
        ShowMesh = kwargs.get('ShowMesh', False)
        NumThreads = kwargs.get('NumThreads',1)

        # MeshCheck routine which allows you to mesh in the GUI (Used for debugging).
        # The script will terminate after this routine
        if MeshCheck and MeshCheck in self.Data.keys():
            self.VL.Logger('### Meshing {} in GUI ###\n'.format(MeshCheck), Print=True)
            # The file MeshParaFile is passed to MeshScript to create the mesh in the GUI
            MeshParaFile = "{}/{}.py".format(self.GEOM_DIR,MeshCheck)
            MeshScript = "{}/{}.py".format(self.VL.SIM_MESH, self.Data[MeshCheck].File)

            SubProc = self.VL.Salome.Run(MeshScript, ArgList=[MeshParaFile], GUI=True)
            SubProc.wait()
            self.VL.Exit('Terminating after checking mesh')

        elif MeshCheck and MeshCheck not in self.Data.keys():
            self.VL.Exit("Error: '{}' specified for MeshCheck is not one of meshes to be created.\n"\
            	         "Meshes to be created are:{}".format(MeshCheck, list(self.Data.keys())))

        self.VL.Logger('\n### Starting Meshing ###\n',Print=True)

        NumMeshes = len(self.Data)
        NumThreads = min(NumThreads,NumMeshes)

        Arg0 = [self]*len(self.Data)
        Arg1 = list(self.Data.keys())

        pool = ProcessPool(nodes=NumThreads)
        pool.map(self.__Run__, Arg1)

        # from pyina.ez_map import ez_map
        # results = ez_map(extRun, Arg0, Arg1, nodes=NumThreads)
        # for res in results: print(res)

        self.VL.Logger('\n### Meshing Completed ###',Print=True)
        if ShowMesh:
            self.VL.Logger("Opening mesh files in Salome",Print=True)
            ArgDict = {name:"{}/{}.med".format(self.VL.MESH_DIR, name) for name in self.Data.keys()}
            Script = '{}/VLPackages/Salome/ShowMesh.py'.format(self.VL.COM_SCRIPTS)
            SubProc = self.VL.Salome.Run(Script, ArgDict=ArgDict, GUI=True)
            SubProc.wait()
            self.VL.Exit("Terminating after mesh viewing")

def extRun(Meta,arg1):
    Mesh.__Run__(Meta,arg1)


# def Test(Meta,MeshName):
# 	script = '{}/VLPackages/Salome/MeshRun.py'.format(Meta.COM_SCRIPTS)
# 	AddPath = [Meta.SIM_MESH, Meta.GEOM_DIR]
# 	ArgDict = {'Name':MeshName,
# 					'MESH_FILE':"{}/{}.med".format(Meta.MESH_DIR, MeshName),
# 					'RCfile':"{}/{}_RC.txt".format(Meta.GEOM_DIR,MeshName)}
# 	if os.path.isfile('{}/config.py'.format(Meta.SIM_MESH)): ArgDict["ConfigFile"] = True
#
# 	import uuid
# 	from subprocess import Popen
# 	portfile = "tmp/{}".format(uuid.uuid4())
# 	SubProc = Popen('{} -t --ns-port-log {}'.format(salome,portfile),shell='TRUE')
# 	SubProc.wait()
# 	with open(portfile,'r') as f:
# 		port = int(f.readline())
# 	SubProc = Popen("{} kill {}".format(salome, port), shell='TRUE')
# 	SubProc.wait()
#
#
# 	Meta.Salome.TestRun(script, AddPath=AddPath, ArgDict=ArgDict)
