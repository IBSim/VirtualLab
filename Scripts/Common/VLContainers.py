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
######################     base module class     ##############################
###############################################################################
class VL_Module(VLSetup):
    def start_module(self):
        #create a tcp socket and wait to receive runs to perform from the server
        ready_msg = {"msg":"Ready","Cont_id":self.Container}
        Utils.send_data(self.tcp_sock, ready_msg)
        while True:
            data = Utils.receive_data(self.tcp_sock)
            if data:
                if data['msg'] == 'Container_runs':
                    self.Logger(f"{data['tool']} container {self.Container} received job list from server.",print=True)
                    full_task_list = data['tasks']
                    self.run_list =full_task_list[str(self.Container)]
                    self.settings_dict = data['settings']
                    self.Settings(**self.settings_dict)
                    break        
        return

###############################################################################
##############################     CIL     ####################################
###############################################################################
class CIL_Setup(VL_Module):
    def __init__(self, Simulation, Project,Cont_id=2):
    	#################################################################
        # import run/setup functions for CIL
        from .VLTypes import CIL as CILFn
        self.CILFn = CILFn
        #perform setup steps that are common to both VL_modules and VL_manger
        self._Common_init(Simulation, Project,Cont_id)
        # Specify default settings
        self.Settings(Mode='H',Launcher='Process',NbJobs=1,
                      InputDir=VLconfig.InputDir, OutputDir=VLconfig.OutputDir,
                      MaterialDir=VLconfig.MaterialsDir,Cleanup=True)

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
        #Utils.Cont_Finished(self.Container)
        print(exitstr)

###############################################################################
################################     GVXR   ###################################
###############################################################################
class GVXR_Setup(VL_Module):
    def __init__(self, Simulation, Project,Cont_id=2):
    	#################################################################
        # import run/setup functions for GVXR
        from .VLTypes import GVXR as GVXRFn
        self.GVXRFn = GVXRFn
        #perform setup steps that are common to both VL_modules and VL_manger
        self._Common_init(Simulation, Project,Cont_id)
        # Specify default settings
        self.Settings(Mode='H',Launcher='Process',NbJobs=1,
                      InputDir=VLconfig.InputDir, OutputDir=VLconfig.OutputDir,
                      MaterialDir=VLconfig.MaterialsDir,Cleanup=True)
        
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
        self.start_module()
        self.GVXRFn.Setup(self,RunGVXR,self.run_list)   

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
