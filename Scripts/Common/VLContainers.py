import sys
sys.dont_write_bytecode=True
import atexit
import copy
import datetime
import json
import os
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
    def __init__(self, Simulation, Project,Cont_id=2):
        #perform setup steps that are common to both VL_modules and VL_manger
        self._Common_init(Simulation, Project,Cont_id)
        # Specify default settings
        self.Settings(Mode='H',Launcher='Process',NbJobs=1,
                      InputDir=VLconfig.InputDir, OutputDir=VLconfig.OutputDir,
                      MaterialDir=VLconfig.MaterialsDir,Cleanup=True)
        self.tcp_sock = Utils.create_tcp_socket()

    def start_module(self):
        #send ready message then wait to receive runs to perform from the server
        ready_msg = {"msg":"Ready","Cont_id":self.Container}
        Utils.send_data(self.tcp_sock, ready_msg)
        while True:
            data = Utils.receive_data(self.tcp_sock)
            if data:
                if data['msg'] == 'Container_runs':
                    self.Logger(f"container {self.Container} received job list from server.",print=True)
                    #full_task_list = data['tasks']
                    self.run_list = data['tasks']
                    self.settings_dict = data['settings']
                    self.Settings(**self.settings_dict)
                    break        
        return

    def filter_runs(self,param_dict,run_ids=None):
        ''' 
        Function to extract subset of runs from a 
        parm_dict.
        
        Inputs:
        - run_ids: list of runs to perform inside the
                   container. If None the code will 
                   perform all the runs defined in 
                   param_dict.

        - param_dict: dictionary generated by create 
                      parameters function. 

        '''
    # if given a subset of runs extract only those runs
        if run_ids:
            all_runs = list(param_dict.keys())
            run_list = [all_runs[i] for i in run_ids]
        else:
            run_list = list(param_dict.keys())

        param_dict= {key: param_dict[key] for key in param_dict.keys() & run_list}
        
        return param_dict

    def _Cleanup(self,Cleanup=True):

        exitstr = '\n#############################\n'\
                    '####### Module Terminated ######\n'\
                    '#############################\n'\
        
        Utils.Cont_Finished(self.Container,self.tcp_sock)
        print(exitstr)

###############################################################################
##############################     CIL     ####################################
###############################################################################
class CIL_Setup(VL_Module):
    def __init__(self, Simulation, Project,Cont_id=2):
        super().__init__(Simulation, Project,Cont_id)
    	#################################################################
        # import run/setup functions for CIL
        from .VLTypes import CIL as CILFn
        self.CILFn = CILFn

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
        self.start_module()
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
        Utils.Cont_Finished(self.Container,self.tcp_sock)
        print(exitstr)

###############################################################################
################################     GVXR   ###################################
###############################################################################
class GVXR_Setup(VL_Module):
    def __init__(self, Simulation, Project,Cont_id=3):
        super().__init__(Simulation, Project,Cont_id)
        # import run/setup functions for GVXR
        from .VLTypes import GVXR as GVXRFn
        self.GVXRFn = GVXRFn

        
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

###############################################################################
########################     Salome/ERMES   ###################################
###############################################################################
class VL_SIM(VL_Module):
    def __init__(self, Simulation, Project,Cont_id=4):
        super().__init__(Simulation, Project,Cont_id)
    	#################################################################
        # import run/setup functions for Salome and ERMES
        from .VLTypes import Sim as SimFn
        self.SimFn = SimFn
        self.VLRoutine_SCRIPTS = "{}/VLRoutines".format(self.COM_SCRIPTS)

    def Parameters(self, Parameters_Master, Parameters_Var=None, RunSim=False, Import=False):
        # Update args with parsed args
        Parameters_Master = self._ParsedArgs.get('Parameters_Master',Parameters_Master)
        Parameters_Var = self._ParsedArgs.get('Parameters_Var',Parameters_Var)
        RunSim = self._ParsedArgs.get('RunSim',RunSim)
        Import = self._ParsedArgs.get('Import',Import)

        # Create variables based on the namespaces (NS) in the Parameters file(s) provided
        VLNamespaces = ['Sim']
        #Note: The call to GetParams converts params_master/var into Namespaces
        # however we need to original strings for passing into other containers.
        # So we will ned to get them here.
        self.Parameters_Master_str = Parameters_Master
        self.Parameters_Var_str = Parameters_Var
        self.GetParams(Parameters_Master, Parameters_Var, VLNamespaces)
        self.start_module()
        #self.SimFn.Setup(self,RunSim,self.run_list)
        self.SimFn.Setup(self,RunSim, Import)

     #Hook for Salome/Ermes       
    def Sim(self,**kwargs):
        kwargs = self._UpdateArgs(kwargs)
        return self.SimFn.Run(self,**kwargs)

    def devSim(self,**kwargs):
        kwargs = self._UpdateArgs(kwargs)
        return self.SimFn.Run(self,**kwargs)

###############################################################################
########################     Code Aster   ###################################
###############################################################################
class VL_Mesh(VL_Module):
    def __init__(self, Simulation, Project,Cont_id=5):
        super().__init__(Simulation, Project,Cont_id)
    	#################################################################
        # import run/setup functions for Code Aster?
        from .VLTypes import Mesh as MeshFn
        self.MeshFn=MeshFn
        self.VLRoutine_SCRIPTS = "{}/VLRoutines".format(self.COM_SCRIPTS)

    def Parameters(self, Parameters_Master, Parameters_Var=None, RunMesh=False, Import=False):
        # Update args with parsed args
        Parameters_Master = self._ParsedArgs.get('Parameters_Master',Parameters_Master)
        Parameters_Var = self._ParsedArgs.get('Parameters_Var',Parameters_Var)
        RunMesh = self._ParsedArgs.get('RunMesh',RunMesh)
        Import = self._ParsedArgs.get('Import',Import)

        # Create variables based on the namespaces (NS) in the Parameters file(s) provided
        VLNamespaces = ['Mesh']
        #Note: The call to GetParams converts params_master/var into Namespaces
        # however we need to original strings for passing into other containers.
        # So we will ned to get them here.
        self.Parameters_Master_str = Parameters_Master
        self.Parameters_Var_str = Parameters_Var
        self.GetParams(Parameters_Master, Parameters_Var, VLNamespaces)
        self.start_module()
        #self.MeshFn.Setup(self,RunSim,self.run_list)
        self.MeshFn.Setup(self,RunMesh, Import)

    #Hooks for Salome/Ermes       
    def Mesh(self,**kwargs):
        kwargs = self._UpdateArgs(kwargs)
        return self.MeshFn.Run(self,**kwargs)

    def devMesh(self,**kwargs):
        kwargs = self._UpdateArgs(kwargs)
        return self.MeshFn.Run(self,**kwargs)

###############################################################################
#########################     Comms Test     ##################################
###############################################################################
class VL_Comms_Test(VL_Module):
    def __init__(self, Simulation, Project,Cont_id=6):
        super().__init__(Simulation, Project,Cont_id)
    	#################################################################
        # import run/setup functions for tests
        from .VLTypes import Comms as TestFn
        self.TestFn = TestFn


    def Parameters(self, Parameters_Master, Parameters_Var=None, RunTest=False):
        # Update args with parsed args
        Parameters_Master = self._ParsedArgs.get('Parameters_Master',Parameters_Master)
        Parameters_Var = self._ParsedArgs.get('Parameters_Var',Parameters_Var)
        RunTest = self._ParsedArgs.get('RunTest',RunTest)
        
        # Create variables based on the namespaces (NS) in the Parameters file(s) provided
        # Note: CIL uses the GVXR namespace since many of the settings overlap.
        VLNamespaces = ['Test']
        #Note: The call to GetParams converts params_master/var into Namespaces
        # however we need to original strings for passing into other containers.
        # So we will ned to get them here.
        self.Parameters_Master_str = Parameters_Master
        self.Parameters_Var_str = Parameters_Var
        self.GetParams(Parameters_Master, Parameters_Var, VLNamespaces)
        self.start_module()
        self.TestFn.Setup(self,RunTest)

    #Hook for CIL       
    def Test_Coms(self,**kwargs):
        kwargs = self._UpdateArgs(kwargs)
        return self.TestFn.Run(self)