#!/usr/bin/env python3

import sys
sys.dont_write_bytecode=True
import datetime
import os
import shutil
import copy
from types import SimpleNamespace as Namespace
from importlib import import_module, reload
import atexit
import json
import numpy as np
import uuid
import VLconfig
from .VLContainer import Container_Utils as Utils
from . import VLFunctions as VLF
#############################################################
# Note: VLSetup is the VLManager in the V2.0 naming scheme. #
#       However, changing it's name would break far to much #
#       legacy code to be worthwhile. So it's easier to     #
#       just live with it for now.                          #
#############################################################
class VLSetup():
    def __init__(self, Simulation, Project,Cont_id=1):
        #####################################################
        # import run/setup functions for curently all but CIL
        from .VLTypes import DA as DAFn
        self.DAFn=DAFn
        #perform setup steps that are common to both VL_modules and VL_manger
        self._Common_init(Simulation, Project,Cont_id)
        self.VLRoutine_SCRIPTS = "{}/VLRoutines".format(self.COM_SCRIPTS)
        self.tcp_sock = Utils.create_tcp_socket()
        # Unique ID
        git_id = self._git()
        self._ID ="{}_{}".format(git_id,self._time)
        # Specify default settings
        self.Settings(Mode='H',Launcher='Process',NbJobs=1,
                      InputDir=VLconfig.InputDir, OutputDir=VLconfig.OutputDir,
                      MaterialDir=VLconfig.MaterialsDir,Max_Containers=1,
                      Cleanup=True)
        ############################################################
        # dynamically create hook functions for all modules based on config file
        ############################################################
        #module_config = self.load_module_config(VLconfig.VL_DIR)
        ############################################################
        data = {"msg":"VirtualLab started","Cont_id":1}
        data_string = json.dumps(data)
        Utils.send_data(self.tcp_sock,data)
        self.Logger('\n############################\n'\
                        '### Launching VirtualLab ###\n'\
                        '############################\n',Print=True)

    def handle_except(self,*args):
        ''' 
        This function stops errors in containers from occurring silently by 
        catching them with sys.excepthook and printing a traceback to the 
        user via VL.Logger. When python errors are raised it normally only
         prints the traceback to the the container not the host. 
         Hence we need to be careful errors are handled appropriately.
        
        If this were not here Cleanup will get run Via AtExit. 
        This sends the Finished message to the server to tell the main 
        thread to close. This is what we want. However, AtExit does not
        distinguish between errors and a normal exit. More annoyingly 
        it returns 0. Thus the server assumes all is well.

        This Function is a bit of a hack but at least it gives 
        you a heads-up that all is not well.
        ''' 
        import traceback as tb
        btrace = ''.join(tb.format_exception(None,None,args[2]))
        errtype = str(args[1])
        errormsg = '\n############################\n'\
                     '###  Error Occurred   ######\n'\
                     '###  in the container ######\n'\
                        f'{errtype}\n'\
                        f'{btrace}'\
                     '############################\n'

        self.Logger(errormsg,Print=True)
        
    def load_module_config(self,vlab_dir):
        ''' Function to get the config for the 
        modules from VL_Modules.yaml file 
        '''
        import yaml
        from pathlib import Path
        vlab_dir = Path(vlab_dir)
        #load module config from yaml_file
        config_file = vlab_dir/'VL_Modules.yaml'
        with open(config_file)as file:
            try:
                config = yaml.safe_load(file)
            except yaml.YAMLError as exception:
                print(exception)
        return config

    def _Common_init(self,Simulation, Project,Cont_id=1):
        '''
        init steps that are common between both VL_manger and VL_modules. 
        These are here since it makes sense to have them in one place and
        save duplicating work.
        '''
        sys.excepthook= self.handle_except
        ########################################################################
    	 # Get parsed args (achieved using the -k flag when launching VirtualLab).
        self._GetParsedArgs()
        # Copy path at the start for MPI to match sys.path
        self._pypath = sys.path.copy()
        self.Container=Cont_id
        # ======================================================================
        # Define variables
        self.Simulation = self._ParsedArgs.get('Simulation',Simulation)
        self.Project = self._ParsedArgs.get('Project',Project)
        # ======================================================================

        # ======================================================================
        # Define path to scripts
        self.COM_SCRIPTS = "{}/Scripts/Common".format(VLconfig.VL_DIR)
        self.SIM_SCRIPTS = "{}/Scripts/{}".format(VLconfig.VL_DIR, self.Simulation)
        # Add these to path
        sys.path = [self.COM_SCRIPTS,self.SIM_SCRIPTS] + sys.path

        #=======================================================================
        # Define & create temporary directory for work to be saved to
        self._TempDir = VLconfig.TEMP_DIR
        # timestamp
        self._time = (datetime.datetime.now()).strftime("%y.%m.%d_%H.%M.%S.%f")

        self.TEMP_DIR = '{}/VL_{}'.format(self._TempDir, self._time)
        try:
            os.makedirs(self.TEMP_DIR)
        except FileExistsError:
            # Unlikely this would happen. Suffix random number to directory name
            self.TEMP_DIR = "{}_{}".format(self.TEMP_DIR,np.random.randint(1000))
            os.makedirs(self.TEMP_DIR)


    def _SetMode(self,Mode='H'):
        Mode = self._ParsedArgs.get('Mode',Mode)
        # ======================================================================
        # Update mode as shorthand version can be given
        if Mode.lower() in ('i', 'interactive'): self.mode = 'Interactive'
        elif Mode.lower() in ('t','terminal'): self.mode = 'Terminal'
        elif Mode.lower() in ('c', 'continuous'): self.mode = 'Continuous'
        elif Mode.lower() in ('h', 'headless'): self.mode = 'Headless'
        else : self.Exit(VLF.ErrorMessage("Mode must be one of; 'Interactive',\
                                      'Terminal','Continuous', 'Headless'"))

    def _SetLauncher(self,Launcher='Process'):
        Launcher = self._ParsedArgs.get('Launcher',Launcher)
        if Launcher.lower() == 'sequential': self._Launcher = 'Sequential'
        elif Launcher.lower() == 'process': self._Launcher = 'Process'
        elif Launcher.lower() == 'mpi': self._Launcher = 'MPI'
        elif Launcher.lower() == 'mpi_worker': self._Launcher = 'MPI_Worker'
        else: self.Exit(VLF.ErrorMessage("Launcher must be one of; 'Sequential',\
                                     'Process', 'MPI'"))

    def _SetNbJobs(self,NbJobs=1):
        NbJobs = self._ParsedArgs.get('NbJobs',NbJobs)
        if type(NbJobs) == int:
            _NbJobs = NbJobs
        elif type(NbJobs) == float:
            if NbJobs.is_integer():
                _NbJobs = NbJobs
            else:
                self.Exit(VLF.ErrorMessage("NbJobs must be an integer"))
        else:
            self.Exit(VLF.ErrorMessage("NbJobs must be an integer"))

        if _NbJobs >= 1:
            self._NbJobs = _NbJobs
        else:
            self.Exit(VLF.ErrorMessage("NbJobs must be positive"))

    def _SetInputDir(self,InputDir):
        InputDir = self._ParsedArgs.get('InputDir',InputDir)
        if not os.path.isdir(InputDir):
            self.Exit(VLF.ErrorMessage("InputDir is not a valid directory"))
        self._InputDir = InputDir
        self.PARAMETERS_DIR = '{}/{}/{}'.format(self._InputDir, self.Simulation, self.Project)

    def _SetOutputDir(self,OutputDir):
        OutputDir = self._ParsedArgs.get('OutputDir',OutputDir)
        self._OutputDir = OutputDir
        self.PROJECT_DIR = '{}/{}/{}'.format(self._OutputDir, self.Simulation, self.Project)

    def _SetMaterialDir(self,MaterialDir):
        if not os.path.isdir(MaterialDir):
            self.Exit(VLF.ErrorMessage("MaterialDir is not a valid directory"))
        MaterialDir = self._ParsedArgs.get('MaterialDir',MaterialDir)
        self.MATERIAL_DIR = MaterialDir

    def _SetCleanup(self,Cleanup=True):
        if not hasattr(self,'_CleanupFlag'): self._CleanupFlag=Cleanup
        else: atexit.unregister(self._Cleanup)
        atexit.register(self._Cleanup,Cleanup)

    def _SetMax_Containers(self,Max_Containers=1,):
        Max_Containers = self._ParsedArgs.get('Max_Containers',Max_Containers)
        if type(Max_Containers) == int:
            _Max_Containers = Max_Containers
        elif type(Max_Containers) == float:
            if Max_Containers.is_integer():
                _Max_Containers = Max_Containers
            else:
                self.Exit(ErrorMessage("Max_Containers must be an integer"))
        else:
            self.Exit(ErrorMessage("Max_Containers must be an integer"))

        if _Max_Containers >= 1:
            self._Max_Containers = _Max_Containers
        else:
            self.Exit(ErrorMessage("Max_Containers must be positive"))

    
    def Settings(self,**kwargs):
        
        Diff = set(kwargs).difference(['Mode','Launcher','NbJobs','Max_Containers','InputDir',
                                    'OutputDir','MaterialDir','Cleanup'])
        if Diff:
            self.Exit(VLF.ErrorMessage("The following are not valid options in Settings:\n{}".format("\n".join(Diff))))

        if 'Mode' in kwargs:
            self._SetMode(kwargs['Mode'])
        if 'Launcher' in kwargs:
            self._SetLauncher(kwargs['Launcher'])
        if 'Max_Containers' in kwargs:
            self._SetMax_Containers(kwargs['Max_Containers'])
        if 'NbJobs' in kwargs:
            self._SetNbJobs(kwargs['NbJobs'])
        if 'Cleanup' in kwargs:
            self._SetCleanup(kwargs['Cleanup'])
        if 'InputDir' in kwargs:
            self._SetInputDir(kwargs['InputDir'])
        if 'OutputDir' in kwargs:
            self._SetOutputDir(kwargs['OutputDir'])
        if 'MaterialDir' in kwargs:
            self._SetMaterialDir(kwargs['MaterialDir'])
        # save settings as a dict here ready to send to containers
        # this saves us pointlessly recreating it later.
        self.settings_dict = kwargs

    def Parameters(self, Parameters_Master, Parameters_Var=None, ParameterArgs=None,
                    RunMesh=True, RunSim=True, RunDA=True,
                    RunVox=True, RunGVXR=True, RunCIL=True,
                    RunTest=True, Import=False):

        # Update args with parsed args
        Parameters_Master = self._ParsedArgs.get('Parameters_Master',Parameters_Master)
        Parameters_Var = self._ParsedArgs.get('Parameters_Var',Parameters_Var)
        RunMesh = self._ParsedArgs.get('RunMesh',RunMesh)
        RunSim = self._ParsedArgs.get('RunSim',RunSim)
        RunDA = self._ParsedArgs.get('RunDA',RunDA)
        RunVox = self._ParsedArgs.get('RunVox',RunVox)
        RunGVXR = self._ParsedArgs.get('RunGVXR',RunGVXR)
        RunCIL = self._ParsedArgs.get('RunCIL',RunCIL)
        RunTest = self._ParsedArgs.get('RunTest',RunTest)
        Import = self._ParsedArgs.get('Import',Import)

        # Create variables based on the namespaces (NS) in the Parameters file(s) provided
        VLNamespaces = ['Mesh','Sim','DA','Vox','GVXR','CIL','Test']

        bool_list = [RunMesh,RunSim,RunDA,RunGVXR,RunCIL,RunVox,RunTest]

        #Note: The call to GetParams converts params_master/var into Namespaces
        # however we need to original strings for passing into other containers.
        # So we will ned to get them here.
        self.Parameters_Master_str = Parameters_Master
        self.Parameters_Var_str = Parameters_Var
        self.GetParams(Parameters_Master, Parameters_Var, VLNamespaces)

        self.DAFn.Setup(self,RunDA, Import)
        # get the number of runs defined in params for each module
        self.Num_runs=self._get_Num_Runs(bool_list,VLNamespaces)
        # get a list of all the containers and the runs they will process for each module
        self.container_list = self._Spread_over_Containers()
        #self.do_Analytics(self,Num_runs) #disabled pending discussion with llion/Rhydian

    def ImportParameters(self, Rel_Parameters,ParameterArgs=None):
        '''
        Rel_Parameters is a file name relative to the Input directory
        '''
        # Strip .py off the end if it's in the name
        if os.path.splitext(Rel_Parameters)[1]=='.py':
            Rel_Parameters = os.path.splitext(Rel_Parameters)[0]

        Abs_Parameters = "{}/{}.py".format(self.PARAMETERS_DIR,Rel_Parameters)
        # Check File exists
        if not os.path.exists(Abs_Parameters):
            message = "The following Parameter file does not exist:\n{}".format(Abs_Parameters)
            self.Exit(VLF.ErrorMessage(message))

        if ParameterArgs !=None:
            # If arguments are provided then it's pickled to a file
            arg_path = "{}/{}.pkl".format(self.TEMP_DIR,uuid.uuid4())
            VLF.WriteArgs(arg_path,ParameterArgs)
            sys.argv.append("ParameterArgs={}".format(arg_path))

        sys.path.insert(0, os.path.dirname(Abs_Parameters))
        Parameters = reload(import_module(os.path.basename(Rel_Parameters)))
        sys.path.pop(0)

        if ParameterArgs != None:
            sys.argv.pop(-1)


        return Parameters

    def GetParams(self, Master, Var, VLTypes,ParameterArgs=None):
        '''Master & Var can be a module, namespace, string or None.
        A string references a file to import from within the input directory.
        '''

        if Master == None and Var == None:
            message = "Both Parameters_Master or Parameters_Var can't be None"
            self.Exit(VLF.ErrorMessage(message))

        # ======================================================================
        # If string, import files
        if type(Master)==str:
            Master = self.ImportParameters(Master,ParameterArgs)
        if type(Var)==str:
            Var = self.ImportParameters(Var,ParameterArgs)

        # ======================================================================
        # Check any of the attributes of NS are included
        if Master != None and not set(Master.__dict__).intersection(VLTypes):
            message = "Parameters_Master contains none of the attributes {}".format(VLTypes)
            self.Exit(VLF.ErrorMessage(message))
        if Var != None and not set(Var.__dict__).intersection(VLTypes):
            message = "Parameters_Var contains none of the attributes {}".format(VLTypes)
            self.Exit(VLF.ErrorMessage(message))

        # ======================================================================
        self.Parameters_Master = Namespace()
        self.Parameters_Var = Namespace()
        for nm in VLTypes:
            master_nm = getattr(Master, nm, None)
            var_nm = getattr(Var, nm, None)
            # ==================================================================
            # Check all in NS have the attribute 'Name'
            if master_nm != None and not hasattr(master_nm,'Name'):
                message = "'{}' does not have the attribute 'Name' in Parameters_Master".format(nm)
                self.Exit(VLF.ErrorMessage(message))
            if master_nm != None and not hasattr(master_nm,'Name'):
                message = "'{}' does not have the attribute 'Name' in Parameters_Var".format(nm)
                self.Exit(VLF.ErrorMessage(message))

            # ==================================================================
            setattr(self.Parameters_Master, nm, master_nm)
            setattr(self.Parameters_Var, nm, var_nm)

    def CreateParameters(self, junk1, junk2, VLType):
        '''
        Create parameter dictionary of attribute VLType using Parameters_Master and Var.
        '''
        # ======================================================================
        # Get VLType from Parameters_Master and _Parameters_Var (if they are defined)
        Master = getattr(self.Parameters_Master, VLType, None)
        Var = getattr(self.Parameters_Var, VLType, None)

        # Check VLType is an appropriate type
        if type(Master) not in (type(None),type(Namespace())):
            print(VLF.WarningMessage("Variable '{}' named in Master but is not a namespace. This may lead yo unexpected results".format(VLType)))
        if type(Var) not in (type(None),type(Namespace())):
            print(VLF.WarningMessage("Variable '{}' named in Var but is not a namespace. This may lead yo unexpected results".format(VLType)))

        # ======================================================================
        # VLType isn't in Master of Var
        if Master==None and Var==None: return {}

        # ======================================================================
        # VLType is in Master but not in Var
        elif Var==None:
            # Check if VLFunctions.Parameters_Var function has been used to create
            # an iterator to vary parameters within master file.
            typelist = [type(val) for val in Master.__dict__.values()]
            if type(iter([])) in typelist:
                # itertor found so we consider this as a varying parameter
                Var = Namespace()
                for key, val in Master.__dict__.items():
                    if type(val) == type(iter([])):
                        setattr(Var,key,list(val))
                # Check that Name is also an iterator
                if not hasattr(Var,'Name'):
                    message = "{}.Name is not an iterable".format(VLType)
                    self.Exit(VLF.ErrorMessage(message))
                # Assign Var to class. Behaviour is the same as if _Parameters_Var
                # file had been used.
                setattr(self.Parameters_Var,VLType,Var)
            else:
                # No iterator, just a single study
                return {Master.Name : Master.__dict__}

        # ======================================================================
        # VLType is in Var

        # Check all entires in Parameters_Var have the same length
        NbNames = len(Var.Name)
        VarNames, NewVals, errVar = [],[],[]
        for VariableName, NewValues in Var.__dict__.items():
            VarNames.append(VariableName)
            NewVals.append(NewValues)
            if len(NewValues) != NbNames:
                errVar.append(VariableName)

        if errVar:
            attrstr = "\n".join(["{}.{}".format(VLType,i) for i in errVar])
            message = "The following attribute(s) have a different number of entries to {0}.Name in Parameters_Var:\n"\
                "{1}\n\nAll attributes of {0} in Parameters_Var must have the same length.".format(VLType,attrstr)
            self.Exit(VLF.ErrorMessage(message))

        # VLType is in Master and Var
        if Master!=None and Var !=None:
            # Check if there are attributes defined in Var which are not in Master
            dfattrs = set(Var.__dict__.keys()) - set(list(Master.__dict__.keys())+['Run'])
            if dfattrs:
                attstr = "\n".join(["{}.{}".format(VLType,i) for i in dfattrs])
                message = "The following attribute(s) are specified in Parameters_Var but not in Parameters_Master:\n"\
                    "{}\n\nThis may lead to unexpected results.".format(attstr)
                print(VLF.WarningMessage(message))

        # ======================================================================
        # Create dictionary for each entry in Parameters_Var
        VarRun = getattr(Var,'Run',[True]*NbNames) # create True list if Run not an attribute of VLType
        ParaDict = {}
        for Name, NewValues, Run in zip(Var.Name,zip(*NewVals),VarRun):
            if not Run: continue
            base = {} if Master==None else copy.deepcopy(Master.__dict__)
            for VariableName, NewValue in zip(VarNames,NewValues):
                base[VariableName]=NewValue
            ParaDict[Name] = base

        return ParaDict

    def _get_Num_Runs(self,Runbools,Namespaces):
        '''
        Function to get the number of runs defined in Params_master/Var for each Namespace.
        this is used to help calculate how many containers to spawn for parallel runs.
        Inputs:
        Runbools - list of bool's  for namespaces to run 
        Namespaces -  list of namespaces
        returns:
        dict of number of runs defined for each namespace or 0 for each that Runbools is set for.
        '''
        num_runs = {}
        
        for I,module in enumerate(Namespaces):
            # special case for CIL since it shares the GVXR namespace
            if module == 'CIL':
                TMPDict = self.CreateParameters(self.Parameters_Master_str, self.Parameters_Var_str,'GVXR')
            else:
                TMPDict = self.CreateParameters(self.Parameters_Master_str, self.Parameters_Var_str,module)
            # if Run is False or Dict is empty add 0 to list 0 instead.
            if not(Runbools[I] and TMPDict):
                num_runs[module] = 0
            else:
                num_runs[module] = len(TMPDict)
        return (num_runs)
        
    def GetFilePath(self, Dirs, file_name, file_ext='py', exit_on_error=True):
        ''' This function will return either the file path if it exists or None.'''
        # ==========================================================================
        # Check file exists
        if type(Dirs) == str: Dirs=[Dirs]
        FilePath = None
        for dir in Dirs:
            _FilePath = "{}/{}.{}".format(dir,file_name,file_ext)
            FileExist = os.path.isfile(_FilePath)
            if FileExist:
                FilePath = _FilePath
                break
        if exit_on_error and FilePath is None:
            self.Exit(VLF.ErrorMessage("The file {}.{} is not in the following directories:\n"\
                    "{}".format(file_name,file_ext,"\n".join(Dirs))))

        return FilePath

    def _Spread_over_Containers(self):
        '''
        Function to generate a dict with a key for each VitualLab module. 
        The values of the Dict are tuples the first index designates the 
        container and the second designates which runs are assigned to 
        that container.
        '''
        from itertools import cycle
        containers={}

        container_ids = [*range(1,self._Max_Containers+1)]
        y=cycle(container_ids)
        for module in self.Num_runs.keys():
            runs = list(range(0, self.Num_runs[module]))
            temp = []
            for i in container_ids:
                tmp =[]
                for j in runs:
                    x = next(y)
                    if x == i:
                        tmp.append(j)
                if tmp:
                    temp.append((i,tmp))
                y=cycle(container_ids)
            if temp:    
                containers[module] = temp
        return containers


    def GetFunction(self, file_path, func_name, exit_on_error=True):
        func = VLF.GetFunc(file_path,func_name)
        if exit_on_error and func is None:
            self.Exit(VLF.ErrorMessage("The function {} is not "\
                    "in {}".format(func_name,file_path)))

        return func
# Call to Run a container
    def Mesh(self,**kwargs):
        
        # if in main contianer submit job request
        return_value=Utils.Spawn_Container(Cont_id=1,Tool="Salome",
        Num_Cont=len(self.container_list['Mesh']),
        Cont_runs=self.container_list['Mesh'],
        Parameters_Master=self.Parameters_Master_str,
        Parameters_Var=self.Parameters_Var_str,
        Project=self.Project,
        Simulation=self.Simulation,
        Settings=self.settings_dict,
        tcp_socket=self.tcp_sock)

        if return_value != '0':
            #an error ocured so exit VirtualLab
            self.Exit("Error Occurred with Mesh")
        return

    # def devMesh(self,**kwargs):
    #     kwargs = self._UpdateArgs(kwargs)
    #     return self.MeshFn.Run(self,**kwargs)
# Call to Run a container
    def Sim(self,**kwargs):
        
        # if in main contianer submit job request
        return_value=Utils.Spawn_Container(Cont_id=1,Tool="Aster",
        Num_Cont=len(self.container_list['Sim']),
        Cont_runs=self.container_list['Sim'],
        Parameters_Master=self.Parameters_Master_str,
        Parameters_Var=self.Parameters_Var_str,
        Project=self.Project,
        Simulation=self.Simulation,
        Settings=self.settings_dict,
        tcp_socket=self.tcp_sock)

        if return_value != '0':
            #an error ocured so exit VirtualLab
            self.Exit("Error Occurred with Sim")
        return

    # def devSim(self,**kwargs):
    #     kwargs = self._UpdateArgs(kwargs)
    #     return self.SimFn.Run(self,**kwargs)

    def DA(self,**kwargs):
        kwargs = self._UpdateArgs(kwargs)
        return self.DAFn.Run(self,**kwargs)

# Call to Run a container
    def Voxelise(self,**kwargs):
        
        # if in main contianer submit job request
        return_value=Utils.Spawn_Container(Cont_id=1,Tool="Vox",
        Num_Cont=len(self.container_list['Vox']),
        Cont_runs=self.container_list['Vox'],
        Parameters_Master=self.Parameters_Master_str,
        Parameters_Var=self.Parameters_Var_str,
        Project=self.Project,
        Simulation=self.Simulation,
        Settings=self.settings_dict,
        tcp_socket=self.tcp_sock)

        if return_value != '0':
            #an error ocured so exit VirtualLab
            self.Exit("Error Occurred with Cad2Vox")
        return

# Call to Run a container for GVXR
    def CT_Scan(self,**kwargs):
        
        # if in main contianer submit job request
        return_value=Utils.Spawn_Container(Cont_id=1,Tool="GVXR",
        Num_Cont=len(self.container_list['GVXR']),
        Cont_runs=self.container_list['GVXR'],
        Parameters_Master=self.Parameters_Master_str,
        Parameters_Var=self.Parameters_Var_str,
        Project=self.Project,
        Simulation=self.Simulation,
        Settings=self.settings_dict,
        tcp_socket=self.tcp_sock)

        if return_value != '0':
            #an error ocured so exit VirtualLab
            self.Exit("Error Occurred with GVXR")
        return
# Call to Run a container for CIL       
    def CT_Recon(self,**kwargs):
        # if in main container submit job request
        return_value=Utils.Spawn_Container(Cont_id=1,Tool="CIL",
        # Note CIL uses GVXR namespace
        Num_Cont=len(self.container_list['GVXR']),
        Cont_runs=self.container_list['GVXR'],
        Parameters_Master=self.Parameters_Master_str,
        Parameters_Var=self.Parameters_Var_str,
        Project=self.Project,
        Simulation=self.Simulation,
        Settings=self.settings_dict,
        tcp_socket=self.tcp_sock)

        if return_value != '0':
            #an error occurred so exit VirtualLab
            self.Exit("Error Occurred with CIL")
        return

    def devDA(self,**kwargs):
        kwargs = self._UpdateArgs(kwargs)
        return self.DAFn.Run(self,**kwargs)
# Call to spawn a minimal container for testing server communications and docker/apptainer
    def Test_Coms(self,**kwargs):
        # if in main container submit job request
        return_value=Utils.Spawn_Container(Cont_id=1,Tool="Test_Comms",
        Num_Cont=len(self.container_list['Test']),
        Cont_runs=self.container_list['Test'],
        Parameters_Master=self.Parameters_Master_str,
        Parameters_Var=self.Parameters_Var_str,
        Project=self.Project,
        Simulation=self.Simulation,
        Settings=self.settings_dict,
        tcp_socket=self.tcp_sock)

    def Logger(self,Text='',**kwargs):
        Prnt = kwargs.get('Print',False)

        if not hasattr(self,'LogFile'):
            print(Text)
            if self.mode in ('Interactive','Terminal'):
                self.LogFile = None
            else:
                self.LogFile = "{}/.log/{}.log".format(self.PROJECT_DIR, self._time)
                os.makedirs(os.path.dirname(self.LogFile), exist_ok=True)
                with open(self.LogFile,'w') as f:
                    f.write(Text)
                # print("Detailed output written to {}".format(self.LogFile))
            return

        if self.mode in ('Interactive','Terminal'):
            print(Text,flush=True)
        else:
            if Prnt: print(Text,flush=True)
            with open(self.LogFile,'a') as f:
                f.write(Text+"\n")

    def Exit(self, mess='', Cleanup=True):
        self._SetCleanup(Cleanup=Cleanup)
        sys.exit(mess)

    def _Cleanup(self,Cleanup=True):
        from . import Analytics
        # Running with base virtualLab so Report overview of VirtualLab usage
        if hasattr(self,'_Analytics') and VLconfig.VL_ANALYTICS=="True":
            Category = "{}_Overview".format(self.Simulation)
            list_AL = ["{}={}".format(key,value) for key,value in self._Analytics.items()]
            Action = "_".join(list_AL)
            Analytics.Run(Category,Action,self._ID)
            
        exitstr = '\n#############################\n'\
                    '### VirtualLab Terminated ###\n'\
                    '#############################\n'\
        
        if not Cleanup:
            exitstr = 'The temp directory {} has not been deleted.\n'.format(self.TEMP_DIR) + exitstr
        elif os.path.isdir(self.TEMP_DIR):
            shutil.rmtree(self.TEMP_DIR)

        print(exitstr)

    def Cleanup(self,KeepDirs=[]):
        print('Cleanup() is depreciated. You can remove this from your script')

    def _GetParsedArgs(self):
        self._ParsedArgs = {}
        for arg in sys.argv[1:]:
            split=arg.split('=')
            if len(split)!=2:
                continue
            var,value = split
            if value=='False':value=False
            elif value=='True':value=True
            elif value=='None':value=None
            elif value.isnumeric():value=int(value)
            else:
                try: value=float(value)
                except: ValueError

            self._ParsedArgs[var]=value

    def _UpdateArgs(self,ArgDict):
        Changes = set(ArgDict).intersection(self._ParsedArgs)
        if not Changes: return ArgDict

        # If some of the arguments have been parsed then they are updated
        for key in Changes:
            ArgDict[key] = self._ParsedArgs[key]
        return ArgDict
        
    def do_Analytics(VL,Dicts):
        ''' 
        Function to analyse usage of VirtualLab to evidence impact for
        use in future research grant applications. Can be turned off via
        VLconfig.py. See Scripts/Common/Analytics.py for more details.
        '''
        import os
        import VLconfig
        if not N: N = VL._NbJobs
        if VLconfig.VL_ANALYTICS=="True":
            from Scripts.Common import Analytics
            # Create dictionary, if one isn't defined already  
            if not hasattr(VL,'_Analytics'): VL._Analytics = {}

            for vltype in Dicts.keys():
                Category = "{}_{}".format(VL.Simulation,vltype)
                Action = "NJob={}_NCore={}_NNode=1".format(Dicts[vltype],N) #send N_containers?
                # ======================================================================
                # Send information about current job
                Analytics.Run(Category,Action,VL._ID)

                # ======================================================================
                # Update Analytics dictionary with new information
                # Add information to dictionary
                if vltype not in VL._Analytics:
                    VL._Analytics[vltype] = Dicts[vltype]
                else:
                    VL._Analytics[vltype] += Dicts[vltype]
            return
        else:
            return
    
    def _git(self):
        version,branch = '<version>','<branch>'
        try:
            from git import Repo
            repo = Repo(VLconfig.VL_DIR)
            sha = repo.head.commit.hexsha
            version = repo.git.rev_parse(sha, short=7)
            branch = repo.active_branch.name
        except :
            pass
        return "{}_{}".format(version,branch)