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
import uuid

import numpy as np

import VLconfig
from . import Analytics
from . import VLFunctions as VLF
from .VLTypes import Mesh as MeshFn, Sim as SimFn, DA as DAFn, Vox as VoxFn

class VLSetup():
    def __init__(self, Simulation, Project):

        # Get parsed args (achieved using the -k flag when launchign VirtualLab).
        self._GetParsedArgs()
        # Copy path at the start for MPI to match sys.path
        self._pypath = sys.path.copy()

        # ======================================================================
        # Define variables
        self.Simulation = self._ParsedArgs.get('Simulation',Simulation)
        self.Project = self._ParsedArgs.get('Project',Project)

        # ======================================================================
        # Specify default settings
        self.Settings(Mode='H',Launcher='Process',NbJobs=1,
                      InputDir=VLconfig.InputDir, OutputDir=VLconfig.OutputDir,
                      MaterialDir=VLconfig.MaterialsDir,Cleanup=True)

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

    def Settings(self,**kwargs):

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

    def Parameters(self, Parameters_Master, Parameters_Var=None, ParameterArgs=None,
                    RunMesh=True, RunSim=True, RunDA=True,
                    RunVox=True, Import=False):

        # Update args with parsed args
        Parameters_Master = self._ParsedArgs.get('Parameters_Master',Parameters_Master)
        Parameters_Var = self._ParsedArgs.get('Parameters_Var',Parameters_Var)
        RunMesh = self._ParsedArgs.get('RunMesh',RunMesh)
        RunSim = self._ParsedArgs.get('RunSim',RunSim)
        RunDA = self._ParsedArgs.get('RunDA',RunDA)
        RunVox = self._ParsedArgs.get('RunVox',RunVox)
        Import = self._ParsedArgs.get('Import',Import)

        # Create variables based on the namespaces (NS) in the Parameters file(s) provided
        VLNamespaces = ['Mesh','Sim','DA','Vox']
        self.GetParams(Parameters_Master, Parameters_Var, VLNamespaces,
                        ParameterArgs=ParameterArgs)



        MeshFn.Setup(self,RunMesh, Import)
        SimFn.Setup(self,RunSim, Import)
        DAFn.Setup(self,RunDA, Import)
        VoxFn.Setup(self,RunVox)

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
            message = "Parameters_Master contains none of the attrbutes {}".format(VLTypes)
            self.Exit(VLF.ErrorMessage(message))
        if Var != None and not set(Var.__dict__).intersection(VLTypes):
            message = "Parameters_Var contains none of the attrbutes {}".format(VLTypes)
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

    def Mesh(self,**kwargs):
        kwargs = self._UpdateArgs(kwargs)
        return MeshFn.Run(self,**kwargs)

    def devMesh(self,**kwargs):
        kwargs = self._UpdateArgs(kwargs)
        return MeshFn.Run(self,**kwargs)

    def Sim(self,**kwargs):
        kwargs = self._UpdateArgs(kwargs)
        return SimFn.Run(self,**kwargs)

    def devSim(self,**kwargs):
        kwargs = self._UpdateArgs(kwargs)
        return SimFn.Run(self,**kwargs)

    def DA(self,**kwargs):
        kwargs = self._UpdateArgs(kwargs)
        return DAFn.Run(self,**kwargs)

#hook in for cad2vox
    def Voxelise(self,**kwargs):
        kwargs = self._UpdateArgs(kwargs)
        return VoxFn.Run(self,**kwargs)

    def devDA(self,**kwargs):
        kwargs = self._UpdateArgs(kwargs)
        return DAFn.Run(self,**kwargs)

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
