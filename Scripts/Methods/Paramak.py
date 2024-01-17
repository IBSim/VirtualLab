import os
import sys

sys.dont_write_bytecode = True
from types import SimpleNamespace as Namespace

import Scripts.Common.VLFunctions as VLF
from Scripts.Common.VLParallel import VLPool
from Scripts.Common.utils import Method_base
from Scripts.Common.VLContainer import Container_Utils as Utils
from Scripts.VLPackages.Paramak.API import Run as Paramak, Dir as ParamakDIR
"""
Paramak - Neutronics-cad analysis
"""

class Method(Method_base):
    def __init__(self, VL):
        super().__init__(VL)  # run __init__ of Method_base
        self.MethodName = "Paramak"
        self.Containers_used = ["Paramak"]
    def Setup(self, VL, ParamakDicts, Import=False):
        # if either ParamakDicts is empty or RunParamak is False we will return
        if not (self.RunFlag and ParamakDicts):
            return
        VL.tmpParamak_DIR = "{}/Paramak".format(VL.TEMP_DIR)
        os.makedirs(VL.tmpParamak_DIR, exist_ok=True)

        FileDict = {}
        for ParamakName, ParaDict in ParamakDicts.items():
            Parameters = Namespace(**ParaDict)
            CALC_DIR = "{}/{}".format(VL.PROJECT_DIR, ParamakName)
           
            # ==========================================================================
            # Create dictionary for each analysis
            ParamakDict = {
                "CALC_DIR": CALC_DIR,
                "Name": ParamakName,
            }
            if VL.mode == "Headless":
                ParamakDict["Headless"] = True
            else:
                ParamakDict["Headless"] = False

            ParamakDict["copper_interlayer_radius"] = Parameters.copper_interlayer_radius
            ParamakDict["Warmour_thickness"] = Parameters.Warmour_thickness
            ParamakDict["Warmour_width"] = Parameters.Warmour_width
            ParamakDict["Warmour_height_lower"] = Parameters.Warmour_height_lower
            ParamakDict["Warmour_height_upper"] = Parameters.Warmour_height_upper
            ParamakDict["pipe_radius"] = Parameters.pipe_radius
            ParamakDict["pipe_thickness"] = Parameters.pipe_thickness
            ParamakDict["copper_interlayer_thickness"] = Parameters.copper_interlayer_thickness
            ParamakDict["pipe_length"] = Parameters.pipe_length
            ParamakDict["pipe_protrusion"] = Parameters.pipe_protrusion
            ParamakDict["cad_output"] = Parameters.dagmc
          
            self.Data[ParamakName] = ParamakDict.copy()
        return
    @staticmethod
    def PoolRun(VL, ParamakDict):
        
       
        func_name = "cadparamak" # function to be executed within container
        file_name = "{}/neutronics_cad.py".format(VL.SIM_PARAMAK) # python file where 'funcname' is located
       
         
        RC = Paramak(file_name, func_name, fnc_kwargs=ParamakDict)
        return RC

    def Run(self, VL):
        if not self.Data:
            return
        VL.Logger("\n### Starting Neutronics_cad ###\n", Print=True)

        Errorfnc = VLPool(VL, self.GetPoolRun(), self.Data)
        if Errorfnc:
            VL.Exit(
               VLF.ErrorMessage(
                   "\nThe following Neutronics_cad routine(s) finished with errors:\n{}".format(Errorfnc)
               ),
               Cleanup=False,
            )

        VL.Logger("\n### Neutronics_cad Complete ###", Print=True)

