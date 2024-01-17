import os
import sys

sys.dont_write_bytecode = True
from types import SimpleNamespace as Namespace

import Scripts.Common.VLFunctions as VLF
from Scripts.Common.VLParallel import VLPool
from Scripts.Common.utils import Method_base
from Scripts.Common.VLContainer import Container_Utils as Utils
from Scripts.VLPackages.Openmc.API import Run as Openmc, Dir as OpenmcDIR
"""
Openmc - Neutronics-simulation analysis
"""

class Method(Method_base):
    def __init__(self, VL):
        super().__init__(VL)  # run __init__ of Method_base
        self.MethodName = "Openmc"
        self.Containers_used = ["Openmc"]
    def Setup(self, VL, OpenmcDicts, Import=False):
        # if either OpenmcDicts is empty or RunOpenmc is False we will return
        if not (self.RunFlag and OpenmcDicts):
            return
        VL.tmpOpenmc_DIR = "{}/Openmc".format(VL.TEMP_DIR)
        os.makedirs(VL.tmpOpenmc_DIR, exist_ok=True)

        FileDict = {}
        for OpenmcName, ParaDict in OpenmcDicts.items():
            Parameters = Namespace(**ParaDict)
            CALC_DIR = "{}/{}".format(VL.PROJECT_DIR, OpenmcName)
           
            # ==========================================================================
            # Create dictionary for each analysis
            OpenmcDict = {
                "CALC_DIR": CALC_DIR,
                "Name": OpenmcName,
            }
            if VL.mode == "Headless":
                OpenmcDict["Headless"] = True
            else:
                OpenmcDict["Headless"] = False

            
            OpenmcDict["Warmour_thickness"] = Parameters.Warmour_thickness
            OpenmcDict["Warmour_width"] = Parameters.Warmour_width
            OpenmcDict["Warmour_height_lower"] = Parameters.Warmour_height_lower
            OpenmcDict["Warmour_height_upper"] = Parameters.Warmour_height_upper
            OpenmcDict["pipe_protrusion"] = Parameters.pipe_protrusion
            OpenmcDict["source_location"] = Parameters.source_location
            OpenmcDict["heat_output"] = Parameters.heat_output
            OpenmcDict["damage_energy_output"] = Parameters.damage_energy_output
            OpenmcDict["cad_input"] = Parameters.dagmc
            OpenmcDict["width"] = Parameters.width
            OpenmcDict["height"] = Parameters.height
            OpenmcDict["thickness"] = Parameters.thickness
            self.Data[OpenmcName] = OpenmcDict.copy()
        return
    @staticmethod
    def PoolRun(VL, OpenmcDict):
        
       
        func_name = "simulation_Openmc" # function to be executed within container
        file_name = "{}/neutronics_simulation.py".format(VL.SIM_OPENMC) # python file where 'funcname' is located
       
         
        RC = Openmc(file_name, func_name, fnc_kwargs=OpenmcDict)
        return RC

    def Run(self, VL):
        if not self.Data:
            return
        VL.Logger("\n### Starting Neutronics_simulation ###\n", Print=True)

        Errorfnc = VLPool(VL, self.GetPoolRun(), self.Data)
        if Errorfnc:
            VL.Exit(
               VLF.ErrorMessage(
                   "\nThe following Neutronics_simulation routine(s) finished with errors:\n{}".format(Errorfnc)
               ),
               Cleanup=False,
            )

        VL.Logger("\n### Neutronics_simulation Complete ###", Print=True)

