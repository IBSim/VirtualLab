
import os
import sys
sys.dont_write_bytecode=True
from types import SimpleNamespace as Namespace
from importlib import import_module
import time
import shutil
import uuid
import traceback


def Setup(VL, **kwargs):
    VL.MESH_DIR = "{}/Meshes".format(VL.PROJECT_DIR)

    VL.MeshData = {}
    MeshDicts = VL.CreateParameters(VL.Parameters_Master, VL.Parameters_Var,'Mesh')

    # if either MeshDicts is empty or RunMesh is False we will return
    if not (kwargs.get('RunMesh', True) and MeshDicts): return

    VL.GEOM_DIR = '{}/Geom'.format(VL.TEMP_DIR)
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

        Mdict = {'Parameters':Namespace(**ParaDict)}
        if VL.mode in ('Headless','Continuous'):
            Mdict['LogFile'] = "{}/{}.log".format(VL.MESH_DIR,MeshName)
        else : Mdict['LogFile'] = None
        VL.MeshData[MeshName] = Mdict.copy()


def PoolRun(VL, MeshDict,**kwargs):
    MeshName = MeshDict['Parameters'].Name
    VL.Logger("{} started".format(MeshName),Print=True)

    # Returner is a namespace to store data to return to VL
    Returner = Namespace(Error=None)
    try:
        script = '{}/VLPackages/Salome/MeshRun.py'.format(VL.COM_SCRIPTS)
        # if MeshRun is Mesh folder this is used instead
        if os.path.isfile('{}/MeshRun.py'.format(VL.SIM_MESH)):
            script = '{}/MeshRun.py'.format(VL.SIM_MESH)

        AddPath = [VL.SIM_MESH, VL.GEOM_DIR]
        Returnfile = "{}/{}_RC.txt".format(VL.GEOM_DIR,MeshName) # File where salome can write an exit status to
        ArgDict = {'Name':MeshName,
                    'MESH_FILE':"{}/{}.med".format(VL.MESH_DIR, MeshName),
                    'RCfile':Returnfile,
                    'STEP':True}

        err = VL.Salome.Run(script, AddPath=AddPath, ArgDict=ArgDict, OutFile=MeshDict['LogFile'])

        if err:
            Returner.Error = "Error in Salome run"

        # if os.path.isfile(Returnfile):
        #     with open(Returnfile,'r') as f:
        #         Returner.Code = int(f.readline())

        if not Returner.Error:
            Message = "{} completed successfully.".format(MeshName)
        else:
            Message = "{} failed. {}.".format(MeshName,Returner.Error)
        if MeshDict['LogFile']:
            Message += " See the output file {} for more details".format(MeshDict['LogFile'])
        VL.Logger(Message,Print=True)

        return Returner

    except:
        exc = traceback.format_exc()
        VL.Logger("{} raised an exception:\n{}".format(MeshName,exc),Print=True)
        Returner.Error = exc
        return Returner


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

    MeshNames = list(VL.MeshData.keys())
    MeshDicts = list(VL.MeshData.values())

    Arg0 = [VL]*len(VL.MeshData)

    launcher = kwargs.get('launcher','Process')
    onall = kwargs.get('onall',True)
    if launcher == 'Process':
        from pathos.multiprocessing import ProcessPool
        pool = ProcessPool(nodes=NumThreads, workdir=VL.TEMP_DIR)
    elif launcher == 'MPI':
        from pyina.launchers import MpiPool
        pool = MpiPool(nodes=NumThreads,source=True, workdir=VL.TEMP_DIR)
    elif launcher == 'Slurm':
        from pyina.launchers import SlurmPool
        pool = SlurmPool(nodes=NumThreads,workdir=VL.TEMP_DIR)

    Res = pool.map(PoolRun, Arg0, MeshDicts, onall=onall)

    MeshError = []
    for Name,Returner in zip(MeshNames,Res):
        if Returner.Error:
            MeshError.append(Name)
        else :
            shutil.copy("{}/{}.py".format(VL.GEOM_DIR,Name), VL.MESH_DIR)

            # if VL.mode not in ('Interactive','Terminal'):
                # shutil.copy()
                # with open(tmpLogstr.format(VL.GEOM_DIR,tmpMeshName),'r') as rtmpLog:
                #     VL.Logger("\nOutput for '{}':\n{}".format(tmpMeshName,rtmpLog.read()))

    if MeshError:
        VL.Exit("\nThe following meshes finished with errors:\n{}".format(MeshError),KeepDirs=['Geom'])

    VL.Logger('\n### Meshing Complete ###',Print=True)


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
        MeshScript = "{}/{}.py".format(VL.SIM_MESH, VL.MeshData[MeshCheck]['Parameters'].File)
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
        VL.Exit("Salome not initiated")
    # Reduce NumThreads if fewer salome sessions have been created than requested
    elif len(Ports) < NumThreads:
        NumThreads=len(Ports)

    # Keep count number of meshes each session has created due to memory leak
    PortCount = {Port:0 for Port in Ports}

    # Script which is used to import the necessary mesh function
    MeshScript = '{}/VLPackages/Salome/MeshRun.py'.format(VL.COM_SCRIPTS)
    # if MeshRun is in Mesh folder this is used instead
    if os.path.isfile('{}/MeshRun.py'.format(VL.SIM_MESH)):
        MeshScript = '{}/MeshRun.py'.format(VL.SIM_MESH)

    AddPath = [VL.SIM_MESH, VL.GEOM_DIR]
    ArgDict = {}

    MeshStat = {}
    NumActive=NumComplete=0
    SalomeReset = 500 #Close Salome session(s) & open new after this many meshes due to memory leak
    for MeshName, MeshDict in VL.MeshData.items():
        MeshPara = MeshDict['Parameters']
        VL.Logger("'{}' started".format(MeshName),Print=True)

        port = Ports.pop(0)

        ArgDict.update(Name=MeshName, MESH_FILE="{}/{}.med".format(VL.MESH_DIR, MeshName),
                       RCfile="{}/{}_RC.txt".format(VL.GEOM_DIR,MeshName))

        Proc = VL.Salome.Shell(MeshScript, Port=port, AddPath=AddPath, ArgDict=ArgDict, OutFile=MeshDict['LogFile'])
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
                    tmpMeshDict = VL.MeshData[tmpMeshName]
                    # Check if any returncode provided
                    RCfile="{}/{}_RC.txt".format(VL.GEOM_DIR,tmpMeshName)
                    if os.path.isfile(RCfile):
                        with open(RCfile,'r') as f:
                            returncode=int(f.readline())
                        AffectedSims = [Name for Name, StudyDict in VL.SimData.items() if StudyDict["Parameters"].Mesh == tmpMeshName]
                        MeshPara = tmpMeshDict['Parameters']
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
                        Message = "'{}' failed. Error in Salome run.".format(tmpMeshName)
                        if tmpMeshDict['LogFile']:
                            Message += " See the output file {} for more details".format(tmpMeshDict['LogFile'])
                        VL.Logger(Message, Print=True)
                        MeshError.append(tmpMeshName)
                    # SubProc returned successfully
                    else:
                        Message = "'{}' completed successfully.".format(tmpMeshName)
                        if tmpMeshDict['LogFile']:
                            Message += " See the output file {} for more details".format(MeshDict['LogFile'])
                        VL.Logger(Message, Print=True)
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
