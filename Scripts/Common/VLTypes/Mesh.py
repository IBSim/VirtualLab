
import os
import sys
sys.dont_write_bytecode=True
from types import SimpleNamespace as Namespace
from importlib import import_module
from pathos.multiprocessing import ProcessPool
import time
import shutil
import uuid
import traceback
from ..VLPackages import Salome

def Setup(VL, **kwargs):
    VL.MESH_DIR = "{}/Meshes".format(VL.PROJECT_DIR)

    VL.MeshData = {}
    MeshDicts = VL.CreateParameters(VL.Parameters_Master, VL.Parameters_Var,'Mesh')

    # if either MeshDicts is empty or RunMesh is False we will return
    if not (kwargs.get('RunMesh', True) and MeshDicts): return

    VL.GEOM_DIR = '{}/Geom'.format(VL.TMP_DIR)
    os.makedirs(VL.GEOM_DIR, exist_ok=True)
    os.makedirs(VL.MESH_DIR, exist_ok=True)

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
    try:
        Returner = Namespace(Error=None)

        script = '{}/VLPackages/Salome/MeshRun.py'.format(VL.COM_SCRIPTS)
        AddPath = [VL.SIM_MESH, VL.GEOM_DIR]
        Returnfile = "{}/{}_RC.txt".format(VL.GEOM_DIR,MeshName) # File where salome can write an exit status to
        ArgDict = {'Name':MeshName,
        				'MESH_FILE':"{}/{}.med".format(VL.MESH_DIR, MeshName),
        				'RCfile':Returnfile,
                        'STEP':True}
        if os.path.isfile('{}/config.py'.format(VL.SIM_MESH)): ArgDict["ConfigFile"] = True

        if VL.mode in ('Interactive','Terminal'): OutFile=None
        else : OutFile = "{}/{}.log".format(VL.MESH_DIR,MeshName)

        err = VL.Salome.Run(script, AddPath=AddPath, ArgDict=ArgDict, OutFile=OutFile)
        if err:
            VL.Logger("Error code {} returned in Salome run".format(err))
            Returner.Error = [err, "Error in Salome run"]

        if os.path.isfile(Returnfile):
            with open(Returnfile,'r') as f:
                Returner.Error = [int(f.readline()), "Salome returned a value"]

        return Returner

    except:
        exc = traceback.format_exc()
        VL.Logger(exc)
        return Namespace(Exception=exc)

def devRun(VL,**kwargs):
    if not VL.MeshData: return

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

    Arg0 = [VL]*len(VL.MeshData)
    Arg1 = list(VL.MeshData.keys())

    launcher = kwargs.get('launcher','Process')
    onall = kwargs.get('onall',True)
    if launcher == 'Process':
        pool = ProcessPool(nodes=NumThreads)
    elif launcher == 'MPI':
        from pyina.launchers import MpiPool
        pool = MpiPool(nodes=NumThreads,source=True)
    elif launcher == 'Slurm':
        from pyina.launchers import SlurmPool
        pool = SlurmPool(nodes=NumThreads)

    Res = pool.map(PoolRun, Arg0, Arg1, onall=onall)

    MeshError = []
    for Name, Returner in zip(Arg1,Res):
        if hasattr(Returner,'Exception'):
            VL.Logger("'{}' threw an exception".format(Name),Print=True)
            MeshError.append(Name)
            continue

        if Returner.Error:
            VL.Logger("'{}' finished with errors".format(Name),Print=True)
            MeshError.append(Name)
        else :
            VL.Logger("'{}' completed successfully".format(Name), Print=True)
            shutil.copy("{}/{}.py".format(VL.GEOM_DIR,Name), VL.MESH_DIR)
            # if VL.mode not in ('Interactive','Terminal'):
                # shutil.copy()
            	# with open(tmpLogstr.format(VL.GEOM_DIR,tmpMeshName),'r') as rtmpLog:
            	# 	VL.Logger("\nOutput for '{}':\n{}".format(tmpMeshName,rtmpLog.read()))

    if MeshError:
        VL.Exit("The following Meshes finished with errors:\n{}".format(MeshError),KeepDirs=['Geom'])

    VL.Logger('\n### Meshing Completed ###',Print=True)


    if ShowMesh:
        VL.Logger("Opening mesh files in Salome",Print=True)
        ArgDict = {name:"{}/{}.med".format(VL.MESH_DIR, name) for name in VL.MeshData.keys()}
        Script = '{}/VLPackages/Salome/ShowMesh.py'.format(VL.COM_SCRIPTS)
        VL.Salome.Run(Script, ArgDict=ArgDict, GUI=True)
        VL.Exit("Terminating after mesh viewing")

def Run(VL, **kwargs):
    if not VL.MeshData: return

    kwargs.update(VL.GetArgParser()) # Update with any kwarg passed in the call
    '''
    kwargs available:
    MeshCheck: input a meshname and it will open this mesh in the GUI
    ShowMesh: Opens up the meshes in the GUI. Boolean
    NumThreads: Number of different meshes to execute concurrently
    '''
    MeshCheck = kwargs.get('MeshCheck', None)
    ShowMesh = kwargs.get('ShowMesh', False)
    NumThreads = kwargs.get('NumThreads',1)

    # MeshCheck routine which allows you to mesh in the GUI (Used for debugging).
    # The script will terminate after this routine
    if MeshCheck and MeshCheck in VL.MeshData.keys():
    	VL.Logger('### Meshing {} in GUI ###\n'.format(MeshCheck), Print=True)

    	MeshParaFile = "{}/{}.py".format(VL.GEOM_DIR,MeshCheck)
    	MeshScript = "{}/{}.py".format(VL.SIM_MESH, VL.MeshData[MeshCheck].File)
    	# The file MeshParaFile is passed to MeshScript to create the mesh in the GUI
    	VL.Salome.Run(MeshScript, ArgList=[MeshParaFile], GUI=True)
    	VL.Exit('Terminating after checking mesh')

    elif MeshCheck and MeshCheck not in VL.MeshData.keys():
    	VL.Exit("Error: '{}' specified for MeshCheck is not one of meshes to be created.\n"\
    			  "Meshes to be created are:{}".format(MeshCheck, list(VL.MeshData.keys())))

    VL.Logger('\n### Starting Meshing ###\n',Print=True)

    NumMeshes = len(VL.MeshData)
    NumThreads = min(NumThreads,NumMeshes)
    MeshError = []

    # Start #NumThreads number of Salome sessions
    Ports = VL.Salome.Start(NumThreads, OutFile=VL.LogFile)
    # Exit if no Salome sessions have been created
    if len(Ports)==0:
    	VL.Exit("Salome not initiated",Print=True)
    # Reduce NumThreads if fewer salome sessions have been created than requested
    elif len(Ports) < NumThreads:
    	NumThreads=len(Ports)

    # Keep count number of meshes each session has created due to memory leak
    PortCount = {Port:0 for Port in Ports}

    # Script which is used to import the necessary mesh function
    MeshScript = '{}/VLPackages/Salome/MeshRun.py'.format(VL.COM_SCRIPTS)
    AddPath = [VL.SIM_MESH, VL.GEOM_DIR]
    ArgDict = {}
    if os.path.isfile('{}/config.py'.format(VL.SIM_MESH)): ArgDict["ConfigFile"] = True

    tmpLogstr = "" if VL.mode in ('Interactive','Terminal') else "{}/{}_log"
    MeshStat = {}
    NumActive=NumComplete=0
    SalomeReset = 500 #Close Salome session(s) & open new after this many meshes due to memory leak
    for MeshName, MeshPara in VL.MeshData.items():
    	VL.Logger("'{}' started".format(MeshName),Print=True)

    	port = Ports.pop(0)
    	tmpLog = tmpLogstr.format(VL.GEOM_DIR,MeshName)

    	ArgDict.update(Name=MeshName, MESH_FILE="{}/{}.med".format(VL.MESH_DIR, MeshName),
    				   RCfile="{}/{}_RC.txt".format(VL.GEOM_DIR,MeshName))

    	Proc = VL.Salome.Shell(MeshScript, Port=port, AddPath=AddPath, ArgDict=ArgDict, OutFile=tmpLog)
    	MeshStat[MeshName] = [Proc,port]
    	PortCount[port] +=1
    	NumActive+=1
    	NumComplete+=1
    	while NumActive==NumThreads or NumComplete==NumMeshes:
    		for tmpMeshName, SalomeInfo in MeshStat.copy().items():
    			Proc, port = SalomeInfo
    			Poll = Proc.poll()
    			# If SubProc finished Poll will change from None to errorcode
    			if Poll is not None:
    				if VL.mode not in ('Interactive','Terminal'):
    					with open(tmpLogstr.format(VL.GEOM_DIR,tmpMeshName),'r') as rtmpLog:
    						VL.Logger("\nOutput for '{}':\n{}".format(tmpMeshName,rtmpLog.read()))

    				# Check if any returncode provided
    				RCfile="{}/{}_RC.txt".format(VL.GEOM_DIR,tmpMeshName)
    				if os.path.isfile(RCfile):
    					with open(RCfile,'r') as f:
    						returncode=int(f.readline())
    					AffectedSims = [Name for Name, StudyDict in VL.SimData.items() if StudyDict["Parameters"].Mesh == tmpMeshName]
    					MeshPara = VL.MeshData[tmpMeshName]
    					MeshImp = import_module('Mesh.{}'.format(MeshPara.File))
    					# Check in meshfile for error code handling
    					if hasattr(MeshImp,'HandleRC'):
    						VL.Logger("'{}'' returned code {}. "\
    									"Passed to HandleRC function.".format(tmpMeshName,returncode),Print=True)
    						MeshImp.HandleRC(returncode,VL.SimData,AffectedSims,tmpMeshName, MeshError)
    					else :
    						VL.Logger("'{}' returned code {}. Added to error list "\
    									"since no HandleRC function found".format(tmpMeshName,returncode),Print=True)
    						MeshError.append(tmpMeshName)
    				# SubProc returned with error code
    				elif Poll != 0:
    					VL.Logger("'{}' finished with errors".format(tmpMeshName), Print=True)
    					MeshError.append(tmpMeshName)
    				# SubProc returned successfully
    				else :
    					VL.Logger("'{}' completed successfully".format(tmpMeshName), Print=True)
    					shutil.copy("{}/{}.py".format(VL.GEOM_DIR,tmpMeshName), VL.MESH_DIR)

    				# Check if a new salome sesion is needed to free up memory
    				# for the next mesh
    				if NumComplete < NumMeshes and PortCount[port] >= SalomeReset/NumThreads:
    					VL.Logger("Limit reached on Salome session {}".format(port))
    					Salome_Close = VL.Salome.Close(port)
    					port = VL.Salome.Start(OutFile=VL.LogFile)[0]
    					PortCount[port] = 0
    					Salome_Close.wait()

    				MeshStat.pop(tmpMeshName)
    				Proc.terminate()
    				NumActive-=1
    				Ports.append(port)

    		time.sleep(0.1)
    		if not len(MeshStat): break

    if MeshError: VL.Exit("The following Meshes finished with errors:\n{}".format(MeshError),KeepDirs=['Geom'])

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
