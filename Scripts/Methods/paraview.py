import os
import sys

sys.dont_write_bytecode = True
from types import SimpleNamespace as Namespace

import Scripts.Common.VLFunctions as VLF
from Scripts.Common.VLParallel import VLPool
from Scripts.Common.utils import Method_base
from Scripts.Common.VLContainer import Container_Utils as Utils
from Scripts.VLPackages.paraview.API import Run as paraview, Dir as paraviewDIR
"""
paraview - Neutronics-postprocess analysis
"""

class Method(Method_base):
    def __init__(self, VL):
        super().__init__(VL)  # run __init__ of Method_base
        self.MethodName = "paraview"
        self.Containers_used = ["paraview"]
    def Setup(self, VL, paraviewDicts, Import=False):
        # if either paraviewDicts is empty or Runparaview is False we will return
        if not (self.RunFlag and paraviewDicts):
            return
        VL.tmpparaview_DIR = "{}/paraview".format(VL.TEMP_DIR)
        os.makedirs(VL.tmpparaview_DIR, exist_ok=True)

        FileDict = {}
        for paraviewName, ParaDict in paraviewDicts.items():
            Parameters = Namespace(**ParaDict)
            CALC_DIR = "{}/{}".format(VL.PROJECT_DIR, paraviewName)
           
            # ==========================================================================
            # Create dictionary for each analysis
            paraviewDict = {
                "CALC_DIR": CALC_DIR,
                "Name": paraviewName,
            }
            if VL.mode == "Headless":
                paraviewDict["Headless"] = True
            else:
                paraviewDict["Headless"] = False

            
            self.Data[paraviewName] = paraviewDict.copy()
        return
    @staticmethod
    def PoolRun(VL, paraviewDict):
        
       
        func_name = "simulation_paraview" # function to be executed within container
        file_name = "{}/neutronics_post.py".format(VL.SIM_PARAVIEW) # python file where 'funcname' is located
       
         
        RC = paraview(file_name, func_name, fnc_kwargs=paraviewDict)
        return RC

    def Run(self, VL):
        if not self.Data:
            return
        VL.Logger("\n### Starting Neutronics_post ###\n", Print=True)

        Errorfnc = VLPool(VL, self.GetPoolRun(), self.Data)
        if Errorfnc:
            VL.Exit(
               VLF.ErrorMessage(
                   "\nThe following Neutronics_post routine(s) finished with errors:\n{}".format(Errorfnc)
               ),
               Cleanup=False,
            )

        VL.Logger("\n### Neutronics_post Complete ###", Print=True)

