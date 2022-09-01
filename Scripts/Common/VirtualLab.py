#!/usr/bin/env python3

import sys
sys.dont_write_bytecode=True
import datetime
import os
import numpy as np
import shutil
import copy
from types import SimpleNamespace as Namespace
from importlib import import_module, reload
import atexit
import json
import VLconfig
from . import Analytics
from .VLFunctions import ErrorMessage, WarningMessage
from .VLContainer import Container_Utils as Utils
class VLSetup():
    def __init__(self, Simulation, Project,Cont_id=1):
        #######################################################################
        # import run/setup functions for curently all but CIL
        from .VLTypes import Mesh as MeshFn, Sim as SimFn, DA as DAFn, \
        Vox as VoxFn, GVXR as GVXRFn
        self.MeshFn=MeshFn
        self.SimFn=SimFn
        self.DAFn=DAFn
        self.VoxFn=VoxFn
        self.GVXRFn=GVXRFn
        ########################################################################
        # Get parsed args (achieved using the -k flag when launchign VirtualLab).
        self._GetParsedArgs()
        # Copy path at the start for MPI to match sys.path
        self._pypath = sys.path.copy()
        self.Container=Cont_id
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
        if self.Container==1:
            # setup networking to comunicate with host script whilst running in a continer
            import socket
            data = {"msg":"VirtualLab started","Cont_id":1}
            data_string = json.dumps(data)
            sock = socket.socket()
            sock.connect(("0.0.0.0", 9999))
            sock.sendall(data_string.encode('utf-8'))
            sock.close()
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
        else : self.Exit(ErrorMessage("Mode must be one of; 'Interactive',\
                                      'Terminal','Continuous', 'Headless'"))

    def _SetLauncher(self,Launcher='Process'):
        Launcher = self._ParsedArgs.get('Launcher',Launcher)
        if Launcher.lower() == 'sequential': self._Launcher = 'Sequential'
        elif Launcher.lower() == 'process': self._Launcher = 'Process'
        elif Launcher.lower() == 'mpi': self._Launcher = 'MPI'
        elif Launcher.lower() == 'mpi_worker': self._Launcher = 'MPI_Worker'
        else: self.Exit(ErrorMessage("Launcher must be one of; 'Sequential',\
                                     'Process', 'MPI'"))

    def _SetNbJobs(self,NbJobs=1):
        NbJobs = self._ParsedArgs.get('NbJobs',NbJobs)
        if type(NbJobs) == int:
            _NbJobs = NbJobs
        elif type(NbJobs) == float:
            if NbJobs.is_integer():
                _NbJobs = NbJobs
            else:
                self.Exit(ErrorMessage("NbJobs must be an integer"))
        else:
            self.Exit(ErrorMessage("NbJobs must be an integer"))

        if _NbJobs >= 1:
            self._NbJobs = _NbJobs
        else:
            self.Exit(ErrorMessage("NbJobs must be positive"))

    def _SetInputDir(self,InputDir):
        InputDir = self._ParsedArgs.get('InputDir',InputDir)
        if not os.path.isdir(InputDir):
            self.Exit(ErrorMessage("InputDir is not a valid directory"))
        self._InputDir = InputDir
        self.PARAMETERS_DIR = '{}/{}/{}'.format(self._InputDir, self.Simulation, self.Project)

    def _SetOutputDir(self,OutputDir):
        OutputDir = self._ParsedArgs.get('OutputDir',OutputDir)
        self._OutputDir = OutputDir
        self.PROJECT_DIR = '{}/{}/{}'.format(self._OutputDir, self.Simulation, self.Project)

    def _SetMaterialDir(self,MaterialDir):
        if not os.path.isdir(MaterialDir):
            self.Exit(ErrorMessage("MaterialDir is not a valid directory"))
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
            self.Exit("Error: {} are not option(s) for settings".format(list(Diff)))

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

    def Parameters(self, Parameters_Master, Parameters_Var=None,
                    RunMesh=True, RunSim=True, RunDA=True, 
                    RunVox=True, RunGVXR=True, RunCIL=False,
                    Import=False):

        # Update args with parsed args
        Parameters_Master = self._ParsedArgs.get('Parameters_Master',Parameters_Master)
        Parameters_Var = self._ParsedArgs.get('Parameters_Var',Parameters_Var)
        RunMesh = self._ParsedArgs.get('RunMesh',RunMesh)
        RunSim = self._ParsedArgs.get('RunSim',RunSim)
        RunDA = self._ParsedArgs.get('RunDA',RunDA)
        RunVox = self._ParsedArgs.get('RunVox',RunVox)
        RunGVXR = self._ParsedArgs.get('RunGVXR',RunGVXR)
        RunCIL = self._ParsedArgs.get('RunCIL',RunCIL)
        Import = self._ParsedArgs.get('Import',Import)

        # Create variables based on the namespaces (NS) in the Parameters file(s) provided
        VLNamespaces = ['Mesh','Sim','DA','Vox','GVXR']
        #Note: The call to GetParams converts params_master/var into Namespaces
        # however we need to origional strings for passing into other containters.
        # So we will ned to get them here.
        self.Parameters_Master_str = Parameters_Master
        self.Parameters_Var_str = Parameters_Var
        self.GetParams(Parameters_Master, Parameters_Var, VLNamespaces)


        self.MeshFn.Setup(self,RunMesh, Import)
        self.SimFn.Setup(self,RunSim, Import)
        self.DAFn.Setup(self,RunDA, Import)
        self.VoxFn.Setup(self,RunVox)
        self.GVXRFn.Setup(self,RunGVXR)
        
    def ImportParameters(self, Rel_Parameters):
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
            self.Exit(ErrorMessage(message))

        sys.path.insert(0, os.path.dirname(Abs_Parameters))
        Parameters = reload(import_module(os.path.basename(Rel_Parameters)))
        sys.path.pop(0)

        return Parameters

    def GetParams(self, Master, Var, VLTypes):
        '''Master & Var can be a module, namespace, string or None.
        A string references a file to import from within the input directory.
        '''

        if Master == None and Var == None:
            message = "Both Parameters_Master or Parameters_Var can't be None"
            self.Exit(ErrorMessage(message))

        # ======================================================================
        # If string, import files
        if type(Master)==str:
            Master = self.ImportParameters(Master)
        if type(Var)==str:
            Var = self.ImportParameters(Var)

        # ======================================================================
        # Check any of the attributes of NS are included
        if Master != None and not set(Master.__dict__).intersection(VLTypes):
            message = "Parameters_Master contains none of the attrbutes {}".format(VLTypes)
            self.Exit(ErrorMessage(message))
        if Var != None and not set(Var.__dict__).intersection(VLTypes):
            message = "Parameters_Var contains none of the attrbutes {}".format(VLTypes)
            self.Exit(ErrorMessage(message))

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
                self.Exit(ErrorMessage(message))
            if master_nm != None and not hasattr(master_nm,'Name'):
                message = "'{}' does not have the attribute 'Name' in Parameters_Var".format(nm)
                self.Exit(ErrorMessage(message))

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

        # ======================================================================
        # VLType isn't in Master of Var
        if Master==None and Var==None: return {}

        # ======================================================================
        # VLType is in Master but not in Var
        elif Var==None: return {Master.Name : Master.__dict__}

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
            self.Exit(ErrorMessage(message))

        # VLType is in Master and Var
        if Master!=None and Var !=None:
            # Check if there are attributes defined in Var which are not in Master
            dfattrs = set(Var.__dict__.keys()) - set(list(Master.__dict__.keys())+['Run'])
            if dfattrs:
                attstr = "\n".join(["{}.{}".format(VLType,i) for i in dfattrs])
                message = "The following attribute(s) are specified in Parameters_Var but not in Parameters_Master:\n"\
                    "{}\n\nThis may lead to unexpected results.".format(attstr)
                print(WarningMessage(message))

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


    def Mesh(self,**kwargs):
        kwargs = self._UpdateArgs(kwargs)
        return self.MeshFn.Run(self,**kwargs)

    def devMesh(self,**kwargs):
        kwargs = self._UpdateArgs(kwargs)
        return self.MeshFn.Run(self,**kwargs)

    def Sim(self,**kwargs):
        kwargs = self._UpdateArgs(kwargs)
        return self.SimFn.Run(self,**kwargs)

    def devSim(self,**kwargs):
        kwargs = self._UpdateArgs(kwargs)
        return self.SimFn.Run(self,**kwargs)

    def DA(self,**kwargs):
        kwargs = self._UpdateArgs(kwargs)
        return self.DAFn.Run(self,**kwargs)

#hook in for cad2vox
    def Voxelise(self,**kwargs):
        kwargs = self._UpdateArgs(kwargs)
        return self.VoxFn.Run(self,**kwargs)
# Hook for GVXR
    def CT_Scan(self,**kwargs):
        kwargs = self._UpdateArgs(kwargs)
        return self.GVXRFn.Run(self,**kwargs)
     #Hook for CIL       
    def CT_Recon(self,**kwargs):
        # if in main contianer submit job request
        Utils.RunJob(Cont_id=1,Tool="CIL",
        Parameters_Master=self.Parameters_Master_str,
        Parameters_Var=self.Parameters_Var_str,
        Project=self.Project,
        Simulation=self.Simulation)
        return
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
            MeshNb = self._Analytics.get('Mesh',0)
            SimNb = self._Analytics.get('Sim',0)
            DANb = self._Analytics.get('DANb',0)
            Category = "{}_Overview".format(self.Simulation)
            Action = "{}_{}_{}".format(MeshNb,SimNb,DANb)
            Analytics.Run(Category,Action,self._ID)
        if self.Container==1:    
            exitstr = '\n#############################\n'\
                        '### VirtualLab Terminated ###\n'\
                        '#############################\n'\
                        
        else:
            Utils.Cont_Finished(self.Container)
            exitstr = '\n#############################\n'\
                        '### Container Terminated ###\n'\
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
        
class CIL_Setup(VLSetup):
    def __init__(self, Simulation, Project,Cont_id=2):
    	#######################################################################
        # import run/setup functions for CIL
        from .VLTypes import CIL as CILFn
        self.CILFn = CILFn
        ########################################################################
    	 # Get parsed args (achieved using the -k flag when launchign VirtualLab).
        self._GetParsedArgs()
        # Copy path at the start for MPI to match sys.path
        self._pypath = sys.path.copy()
        self.Container=Cont_id
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
        
    def Parameters(self, Parameters_Master, Parameters_Var=None, RunCIL=False):
        # Update args with parsed args
        Parameters_Master = self._ParsedArgs.get('Parameters_Master',Parameters_Master)
        Parameters_Var = self._ParsedArgs.get('Parameters_Var',Parameters_Var)
        RunCIL = self._ParsedArgs.get('RunCIL',RunCIL)
        
        # Create variables based on the namespaces (NS) in the Parameters file(s) provided
        VLNamespaces = ['Mesh','Sim','DA','Vox','GVXR']
        #Note: The call to GetParams converts params_master/var into Namespaces
        # however we need to origional strings for passing into other containters.
        # So we will ned to get them here.
        self.Parameters_Master_str = Parameters_Master
        self.Parameters_Var_str = Parameters_Var
        self.GetParams(Parameters_Master, Parameters_Var, VLNamespaces)

        CILFn.Setup(self,RunCIL)
        
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

     #Hook for CIL       
    def CT_Recon(self,**kwargs):
        kwargs = self._UpdateArgs(kwargs)
        return CILFn.Run(self,**kwargs)
