import os
import sys

sys.dont_write_bytecode = True
from types import SimpleNamespace as Namespace

import Scripts.Common.VLFunctions as VLF
from Scripts.Common.VLParallel import VLPool
from Scripts.Common.utils import Method_base
from Scripts.Common.VLContainer import Container_Utils as Utils
from Scripts.VLPackages.paraview.API import Run as lifecycle, Dir as lifecycleDIR
"""
lifecycle - analysis
"""

class Method(Method_base):
    def __init__(self, VL):
        super().__init__(VL)  # run __init__ of Method_base
        self.MethodName = "lifecycle"
        self.Containers_used = ["paraview"]
    def Setup(self, VL, lifecycleDicts, Import=False):
        # if either lifecycleDicts is empty or Runlifecycle is False we will return
        if not (self.RunFlag and lifecycleDicts):
            return
        VL.tmplifecycle_DIR = "{}/lifecycle".format(VL.TEMP_DIR)
        os.makedirs(VL.tmplifecycle_DIR, exist_ok=True)

        FileDict = {}
        for lifecycleName, ParaDict in lifecycleDicts.items():
            Parameters = Namespace(**ParaDict)
            CALC_DIR = "{}/{}".format(VL.PROJECT_DIR, lifecycleName)
           
            # ==========================================================================
            # Create dictionary for each analysis
            lifecycleDict = {
                "CALC_DIR": CALC_DIR,
                "Name": lifecycleName,
            }
            if VL.mode == "Headless":
                lifecycleDict["Headless"] = True
            else:
                lifecycleDict["Headless"] = False

            
            self.Data[lifecycleName] = lifecycleDict.copy()
        return
    @staticmethod
    def PoolRun(VL, lifecycleDict):
        
       
        func_name = "lifecycle_paraview" # function to be executed within container
        file_name = "{}/lifecycle_post.py".format(VL.SIM_LIFECYCLE) # python file where 'funcname' is located
       
         
        RC = lifecycle(file_name, func_name, fnc_kwargs=lifecycleDict)
        return RC

    def Run(self, VL):
        if not self.Data:
            return
        VL.Logger("\n### Starting lifecycle analysis ###\n", Print=True)

        Errorfnc = VLPool(VL, self.GetPoolRun(), self.Data)
        if Errorfnc:
            VL.Exit(
               VLF.ErrorMessage(
                   "\nThe following lifecycle analysis routine(s) finished with errors:\n{}".format(Errorfnc)
               ),
               Cleanup=False,
            )

        VL.Logger("\n### lifecycle analysis Complete ###", Print=True)

