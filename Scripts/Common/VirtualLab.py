#!/usr/bin/env python3

import sys
sys.dont_write_bytecode=True
import datetime
import os
import shutil
import copy
from types import SimpleNamespace as Namespace, ModuleType
import atexit
import uuid

import numpy as np

import VLconfig
from . import Analytics
from . import VLFunctions as VLF
from . import VLTypes

class VLSetup():
    def __init__(self, Simulation, Project):

        # Copy path at the start for MPI to match sys.path
        self._pypath = sys.path.copy()

        # ======================================================================
        # Define variables
        # check for updated to Simulation and Project in parsed arguments
        arg_dict = VLF.Parser_update(Simulation=Simulation,Project=Project)
        self.Simulation = arg_dict['Simulation']
        self.Project = arg_dict['Project']

        # ======================================================================
        # Specify default settings
        self.Settings(Mode='H',Launcher='Process',NbJobs=1,
                      InputDir=VLconfig.InputDir, OutputDir=VLconfig.OutputDir,
                      MaterialDir=VLconfig.MaterialsDir,Cleanup=True)

        # ======================================================================
        # Get the available VLTypes
        self.VLTypes = []
        for attr_name in dir(VLTypes):
            if attr_name.startswith('__'): continue
            vltype_mod = getattr(VLTypes,attr_name)
            self.VLTypes.append(attr_name)

            # set attr_name to be run from the VLType
            run_fn = getattr(vltype_mod,'Run')
            setattr(self,attr_name,_Runner(self,run_fn))


        # ======================================================================
        # Define path to scripts
        self.COM_SCRIPTS = "{}/Scripts/Common".format(VLconfig.VL_DIR)
        self.SIM_SCRIPTS = "{}/Scripts/{}".format(VLconfig.VL_DIR, self.Simulation)
        self.VLRoutine_SCRIPTS = "{}/VLRoutines".format(self.COM_SCRIPTS)
        # Add these to path
        sys.path = [self.COM_SCRIPTS,self.SIM_SCRIPTS] + sys.path

        #=======================================================================
        # Define & create temporary directory for work to be saved to
        self._TempDir = VLconfig.TEMP_DIR
        # timestamp
        self._time = (datetime.datetime.now()).strftime("%y.%m.%d_%H.%M.%S.%f")
        # Unique ID
        stream = os.popen("cd {};git show --oneline -s;git rev-parse --abbrev-ref HEAD".format(VLconfig.VL_DIR))
        output = stream.readlines()
        ver,branch = output[0].split()[0],output[1].strip()
        self._ID ="{}_{}_{}".format(ver,branch,self._time)

        self.TEMP_DIR = '{}/VL_{}'.format(self._TempDir, self._time)
        try:
            os.makedirs(self.TEMP_DIR)
        except FileExistsError:
            # Unlikely this would happen. Suffix random number to direcory name
            self.TEMP_DIR = "{}_{}".format(self.TEMP_DIR,np.random.random_integer(1000))
            os.makedirs(self.TEMP_DIR)

        self.Logger('\n############################\n'\
                      '### Launching VirtualLab ###\n'\
                      '############################\n',Print=True)


    def _SetMode(self,Mode='H'):
        # ======================================================================
        # Update mode as shorthand version can be given
        if Mode.lower() in ('i', 'interactive'): self.mode = 'Interactive'
        elif Mode.lower() in ('t','terminal'): self.mode = 'Terminal'
        elif Mode.lower() in ('c', 'continuous'): self.mode = 'Continuous'
        elif Mode.lower() in ('h', 'headless'): self.mode = 'Headless'
        else : self.Exit(VLF.ErrorMessage("Mode must be one of; 'Interactive',\
                                      'Terminal','Continuous', 'Headless'"))

    def _SetLauncher(self,Launcher='Process'):
        if Launcher.lower() == 'sequential': self._Launcher = 'Sequential'
        elif Launcher.lower() == 'process': self._Launcher = 'Process'
        elif Launcher.lower() == 'mpi': self._Launcher = 'MPI'
        elif Launcher.lower() == 'mpi_worker': self._Launcher = 'MPI_Worker'
        else: self.Exit(VLF.ErrorMessage("Launcher must be one of; 'Sequential',\
                                     'Process', 'MPI'"))

    def _SetNbJobs(self,NbJobs=1):
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
        if not os.path.isdir(InputDir):
            self.Exit(VLF.ErrorMessage("InputDir is not a valid directory"))
        self._InputDir = InputDir
        self.PARAMETERS_DIR = '{}/{}/{}'.format(self._InputDir, self.Simulation, self.Project)

    def _SetOutputDir(self,OutputDir):
        self._OutputDir = OutputDir
        self.PROJECT_DIR = '{}/{}/{}'.format(self._OutputDir, self.Simulation, self.Project)

    def _SetMaterialDir(self,MaterialDir):
        if not os.path.isdir(MaterialDir):
            self.Exit(VLF.ErrorMessage("MaterialDir is not a valid directory"))
        self.MATERIAL_DIR = MaterialDir

    def _SetCleanup(self,Cleanup=True):
        if not hasattr(self,'_CleanupFlag'): self._CleanupFlag=Cleanup
        else: atexit.unregister(self._Cleanup)
        atexit.register(self._Cleanup,Cleanup)

    @VLF.kwarg_update
    def Settings(self,**kwargs):
        # Dont specify the kwarsg so that the efauls aren't overwritten if there
        # are multiple calls to settings
        Diff = set(kwargs).difference(['Mode','Launcher','NbJobs','InputDir',
                                    'OutputDir','MaterialDir','Cleanup'])
        if Diff:
            self.Exit(VLF.ErrorMessage("The following are not valid options in Settings:\n{}".format("\n".join(Diff))))

        if 'Mode' in kwargs:
            self._SetMode(kwargs['Mode'])
        if 'Launcher' in kwargs:
            self._SetLauncher(kwargs['Launcher'])
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

    @VLF.kwarg_update
    def Parameters(self, Parameters_Master, Parameters_Var=None, ParameterArgs=None,
                    RunMesh=True, RunSim=True, RunDA=True,
                    RunVoxelise=True, Import=False):

        # update Parameters_master with parser (not covered by decorator)
        arg_dict = VLF.Parser_update(Parameters_Master=Parameters_Master)
        Parameters_Master = arg_dict['Parameters_Master']

        #
        self._SetParams(Parameters_Master, Parameters_Var,
                       ParameterArgs=ParameterArgs)

        # Run setup for each function
        for vltype in self.VLTypes:
            vltype_mod = getattr(VLTypes,vltype)
            setup_fn = getattr(vltype_mod,'Setup')
            Runflag = locals().get('Run{}'.format(vltype),True)
            setup_fn(self,Runflag)

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

        Parameters = VLF.GetModule(Abs_Parameters)

        if ParameterArgs != None:
            sys.argv.pop(-1)

        return Parameters

    def _SetParams(self, Master, Var, ParameterArgs=None):
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
        # Perform checks of master and var and assign to self
        self.Parameters_Master = self._CheckParams(Master,'Parameters_Master',self.VLTypes)
        self.Parameters_Var = self._CheckParams(Var,'Parameters_Var',self.VLTypes)

    def _CheckParams(self,input,input_name,VLTypes):
        '''Perform checks on the input file and return namespace whose attributes
           are the VLTypes'''
        # if the input is None return empty namespace
        NS = Namespace()
        if input is None: return NS

        # check some of the VLTypes are defined in the input
        if not set(input.__dict__).intersection(VLTypes):
            message = "{} contains none of the attrbutes {}".format(input_name,VLTypes)
            self.Exit(VLF.ErrorMessage(message))

        for nm in VLTypes:
            attr_ns = getattr(input,nm,None) # get attribute nm from input

            # ignore if nm not an attribute
            if attr_ns is None: continue

            # give warning about it not being a module/namespace
            if type(attr_ns) not in (Namespace,ModuleType):
                message = "{} has attribute '{}' but it not a module or namespace.\nThis may lead to unexpected results".format(input_name,nm)
                print(VLF.WarningMessage(message))
                continue

            # check it has a name associated with it
            if not hasattr(attr_ns,'Name'):
                message = "'{}' does not have the attribute 'Name' in {}".format(nm,input_name)
                self.Exit(VLF.ErrorMessage(message))

            setattr(NS,nm,attr_ns) # add the info to the namespace

        return NS

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

    def GetFunction(self, file_path, func_name, exit_on_error=True):
        func = VLF.GetFunc(file_path,func_name)

        if exit_on_error and func is None:
            self.Exit(VLF.ErrorMessage("The function {} is not "\
                    "in {}".format(func_name,file_path)))
        return func

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
        # Report overview of VirtualLab usage
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

# function wrapper for vltypes
def _Runner(VL,func):
    func = VLF.kwarg_update(func) # update kwargs
    def _Runner_wrapper(*args,**kwargs):
        return func(VL,*args,**kwargs)
    return _Runner_wrapper
