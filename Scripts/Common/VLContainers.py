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
from .VirtualLab import VLSetup
###############################################################################
##############################     CIL     ####################################
###############################################################################
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
        # Note: CIL uses the GVXR namespace since many of the settings overlap.
        VLNamespaces = ['GVXR']
        #Note: The call to GetParams converts params_master/var into Namespaces
        # however we need to origional strings for passing into other containters.
        # So we will ned to get them here.
        self.Parameters_Master_str = Parameters_Master
        self.Parameters_Var_str = Parameters_Var
        self.GetParams(Parameters_Master, Parameters_Var, VLNamespaces)

        self.CILFn.Setup(self,RunCIL)
        
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

        print(exitstr)

###############################################################################
################################     GVXR   ###################################
###############################################################################
class GVXR_Setup(VLSetup):
    def __init__(self, Simulation, Project,Cont_id=2):
    	#######################################################################
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
            self.TEMP_DIR = "{}_{}".format(self.TEMP_DIR,np.random.random_integer(1000))
            os.makedirs(self.TEMP_DIR)
        
    def Parameters(self, Parameters_Master, Parameters_Var=None, RunGVXR=False):
        # Update args with parsed args
        Parameters_Master = self._ParsedArgs.get('Parameters_Master',Parameters_Master)
        Parameters_Var = self._ParsedArgs.get('Parameters_Var',Parameters_Var)
        RunGVXR = self._ParsedArgs.get('RunGVXR',RunGVXR)
        
        # Create variables based on the namespaces (NS) in the Parameters file(s) provided
        VLNamespaces = ['GVXR']
        #Note: The call to GetParams converts params_master/var into Namespaces
        # however we need to origional strings for passing into other containters.
        # So we will ned to get them here.
        self.Parameters_Master_str = Parameters_Master
        self.Parameters_Var_str = Parameters_Var
        self.GetParams(Parameters_Master, Parameters_Var, VLNamespaces)

        self.GVXRFn.Setup(self,RunGVXR)
        

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

        print(exitstr)
