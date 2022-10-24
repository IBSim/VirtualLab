import sys
sys.dont_write_bytecode=True
import atexit
import copy
import datetime
import json
import os
import numpy as np
import shutil
from types import SimpleNamespace as Namespace
from importlib import import_module, reload
import VLconfig
from .VirtualLab import VLSetup
from .VLContainer import Container_Utils as Utils

###############################################################################
##############################     CIL     ####################################
###############################################################################
class CIL_Setup(VLSetup):
    def __init__(self, Simulation, Project,Cont_id=2):
    	#################################################################
        # import run/setup functions for CIL
        from .VLTypes import CIL as CILFn
        self.CILFn = CILFn
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

        self.TEMP_DIR = '{}/VL_{}'.format(self._TempDir, self._time)
        try:
            os.makedirs(self.TEMP_DIR)
        except FileExistsError:
            # Unlikely this would happen. Suffix random number to direcory name
            self.TEMP_DIR = "{}_{}".format(self.TEMP_DIR,np.random.randint(1000))
            os.makedirs(self.TEMP_DIR)
        
    def Parameters(self, Parameters_Master, Parameters_Var=None, RunCIL=False):
        # Update args with parsed args
        Parameters_Master = self._ParsedArgs.get('Parameters_Master',Parameters_Master)
        Parameters_Var = self._ParsedArgs.get('Parameters_Var',Parameters_Var)
        RunCIL = self._ParsedArgs.get('RunCIL',RunCIL)
        
        # Create variables based on the namespaces (NS) in the Parameters file(s) provided
        # Note: CIL uses the GVXR namespace since many of the settings overlap.
        VLNamespaces = ['GVXR']
        #Note: The call to GetParams converts params_master/var into Namespaces
        # however we need to original strings for passing into other containers.
        # So we will ned to get them here.
        self.Parameters_Master_str = Parameters_Master
        self.Parameters_Var_str = Parameters_Var
        self.GetParams(Parameters_Master, Parameters_Var, VLNamespaces)
        #create a tcp socket and wait to receive runs to preform from the server 
        tcp_sock = Utils.create_tcp_socket()
        while True:
            data = Utils.receive_data(tcp_sock)
            if data:
                if data['msg'] == 'Container_runs':
                    self.Logger(f"CIL container {self.Container} received job list from server.",print=True)
                    full_task_list = data['tasks']
                    run_list =full_task_list[str(self.Container)] 
                    break
        
        self.CILFn.Setup(self,RunCIL,run_list)
        
    def Settings(self, **kwargs):
        filename = f'{VLconfig.VL_DIR}/Container_settings.json'
        base_settings = self.SettingsFromFile(filename)
        
        # Python merge operator (introduced in python 3.9)
        # Note: this overwrites values in the left dictionary
        # with the value of the keys of the right dictionary,
        # if an overlap exists. This is intentional as it
        # allows values that were passed in to the function to
        # Overwrite the base setings.

        settings = base_settings | kwargs 

        if 'Mode' in settings:
            self._SetMode(settings['Mode'])
        if 'Launcher' in settings:
            self._SetLauncher(settings['Launcher'])
        if 'NbJobs' in settings:
            self._SetNbJobs(settings['NbJobs'])
        if 'Cleanup' in settings:
            self._SetCleanup(settings['Cleanup'])
        if 'InputDir' in settings:
            self._SetInputDir(settings['InputDir'])
        if 'OutputDir' in settings:
            self._SetOutputDir(settings['OutputDir'])
        if 'MaterialDir' in settings:
            self._SetMaterialDir(settings['MaterialDir'])
     #Hook for CIL       
    def CT_Recon(self,**kwargs):
        kwargs = self._UpdateArgs(kwargs)
        return self.CILFn.Run(self,**kwargs)

    def _Cleanup(self,Cleanup=True):

        exitstr = '\n#############################\n'\
                    '####### CIL Terminated ######\n'\
                    '#############################\n'\
        
        if not Cleanup:
            exitstr = 'The temp directory {} has not been deleted.\n'.format(self.TEMP_DIR) + exitstr
        elif os.path.isdir(self.TEMP_DIR):
            shutil.rmtree(self.TEMP_DIR)
        Utils.Cont_Finished(self.Container)
        print(exitstr)

###############################################################################
################################     GVXR   ###################################
###############################################################################
class GVXR_Setup(VLSetup):
    def __init__(self, Simulation, Project,Cont_id=2):
    	#################################################################
        # import run/setup functions for GVXR
        from .VLTypes import GVXR as GVXRFn
        self.GVXRFn = GVXRFn
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

        self.TEMP_DIR = '{}/VL_{}'.format(self._TempDir, self._time)
        try:
            os.makedirs(self.TEMP_DIR)
        except FileExistsError:
            # Unlikely this would happen. Suffix random number to direcory name
            self.TEMP_DIR = "{}_{}".format(self.TEMP_DIR,np.random.randint(1000))
            os.makedirs(self.TEMP_DIR)
        
    def Parameters(self, Parameters_Master, Parameters_Var=None, RunGVXR=False):
        # Update args with parsed args
        Parameters_Master = self._ParsedArgs.get('Parameters_Master',Parameters_Master)
        Parameters_Var = self._ParsedArgs.get('Parameters_Var',Parameters_Var)
        RunGVXR = self._ParsedArgs.get('RunGVXR',RunGVXR)
        
        # Create variables based on the namespaces (NS) in the Parameters file(s) provided
        VLNamespaces = ['GVXR']
        #Note: The call to GetParams converts params_master/var into Namespaces
        # however we need to original strings for passing into other containers.
        # So we will ned to get them here.
        self.Parameters_Master_str = Parameters_Master
        self.Parameters_Var_str = Parameters_Var
        self.GetParams(Parameters_Master, Parameters_Var, VLNamespaces)
        #create a tcp socket and wait to receive runs to perform from the server 
        tcp_sock = Utils.create_tcp_socket()
        ready_msg = {"msg":"Ready","Cont_id":self.Container}
        Utils.send_data(tcp_sock, ready_msg)
        while True:
            data = Utils.receive_data(tcp_sock)
            if data:
                print(data)
                if data['msg'] == 'Container_runs':
                    self.Logger(f"GVXR container {self.Container} received job list from server.",print=True)
                    full_task_list = data['tasks']
                    run_list =full_task_list[str(self.Container)] 
                    break

        self.GVXRFn.Setup(self,RunGVXR,run_list)
        
    def Settings(self, **kwargs):
        filename = f'{VLconfig.VL_DIR}/Container_settings.json'
        base_settings = self.SettingsFromFile(filename)
        
        # Python merge operator (introduced in python 3.9)
        # Note: this overwrites values in the left dictionary
        # with the value of the keys of the right dictionary,
        # if an overlap exists. This is intentional as it
        # allows values that were passed in to the function to
        # Overwrite the base setings.

        settings = base_settings | kwargs 
        if 'Mode' in settings:
            self._SetMode(settings['Mode'])
        if 'Launcher' in settings:
            self._SetLauncher(settings['Launcher'])
        if 'NbJobs' in settings:
            self._SetNbJobs(settings['NbJobs'])
        if 'Cleanup' in settings:
            self._SetCleanup(settings['Cleanup'])
        if 'InputDir' in settings:
            self._SetInputDir(settings['InputDir'])
        if 'OutputDir' in settings:
            self._SetOutputDir(settings['OutputDir'])
        if 'MaterialDir' in settings:
            self._SetMaterialDir(settings['MaterialDir'])    

     #Hook for GVXR       
    def CT_Scan(self,**kwargs):
        kwargs = self._UpdateArgs(kwargs)
        return self.GVXRFn.Run(self,**kwargs)

    def _Cleanup(self,Cleanup=True):

        exitstr = '\n#############################\n'\
                    '###### GVXR Terminated ######\n'\
                    '#############################\n'\
        
        if not Cleanup:
            exitstr = 'The temp directory {} has not been deleted.\n'.format(self.TEMP_DIR) + exitstr
        elif os.path.isdir(self.TEMP_DIR):
            shutil.rmtree(self.TEMP_DIR)
        Utils.Cont_Finished(self.Container)
        print(exitstr)
