import os
import sys

sys.dont_write_bytecode = True
from types import SimpleNamespace as Namespace

import Scripts.Common.VLFunctions as VLF
from Scripts.Common.VLParallel import VLPool
from Scripts.Common.utils import Method_base
from Scripts.Common.VLContainer import Container_Utils as Utils

"""
DPA - DPA Analysis
"""


class Method(Method_base):
    def __init__(self, VL):
        super().__init__(VL)  # run __init__ of Method_base
        self.MethodName = "DPA"

    def Setup(self, VL, DPADicts, Import=False):
        # if either DPADicts is empty or RunDPA is False we will return
        if not (self.RunFlag and DPADicts):
            return

        VL.tmpDPA_DIR = "{}/DPA".format(VL.TEMP_DIR)
        os.makedirs(VL.tmpDPA_DIR, exist_ok=True)

        FileDict = {}
        for DPAName, ParaDict in DPADicts.items():
            CALC_DIR = "{}/{}".format(VL.PROJECT_DIR, DPAName)
            if Import:
                ParaDict = VLF.ImportUpdate(
                    "{}/Parameters.py".format(CALC_DIR), ParaDict
                )

            # ======================================================================
            # get file path & perform checks
            # default name is Single
            file_name, func_name = VLF.FileFuncSplit(ParaDict["File"], "dpa_calculation")

            if (file_name, func_name) not in FileDict:
                # Check file in directory & get path
                FilePath = VLF.GetFilePath(
                    [VL.SIM_DPA, VL.VLRoutine_SCRIPTS],
                    file_name,
                    file_ext="py",
                    exit_on_error=True,
                )
                # Check function func_name is in the file
                VLF.GetFunction(FilePath, func_name, exit_on_error=True)
                File_func = [FilePath, func_name]
                FileDict[(file_name, func_name)] = File_func
            else:
                File_func = FileDict[(file_name, func_name)]

            # ==========================================================================
            # Create dictionary for each analysis
            DPADict = {
                "CALC_DIR": CALC_DIR,
                "TMP_CALC_DIR": "{}/{}".format(VL.tmpDPA_DIR, DPAName),
                "Parameters": Namespace(**ParaDict),
                "FileInfo": File_func,
                "Data": {},
            }

            # Important information can be added to Data during any stage of the
            # data analysis, and this will be saved to the location specified by the
            # value for the __file__ key
            DPADict["Data"] = {"__file__": "{}/Data.pkl".format(DPADict["CALC_DIR"])}

            if VL.mode in ("Headless", "Continuous"):
                DPADict["LogFile"] = "{}/Output.log".format(DPADict["CALC_DIR"])
            else:
                DPADict["LogFile"] = None

            os.makedirs(DPADict["TMP_CALC_DIR"], exist_ok=True)

            self.Data[DPAName] = DPADict

    @staticmethod
    def PoolRun(VL, DPADict):

        Parameters = DPADict["Parameters"]

        os.makedirs(DPADict["CALC_DIR"], exist_ok=True)
        VLF.WriteData("{}/Parameters.py".format(DPADict["CALC_DIR"]), Parameters)

        DPASgl = VLF.GetFunc(*DPADict["FileInfo"])

        err = DPASgl(VL, DPADict)

        return err

    def Run(self, VL):
        VL.AddToPath(VL.SIM_DPA,0)

        VL.Logger(
            "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
            "~~~ Starting DPA Analysis ~~~\n"
            "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n",
            Print=True,
        )

        Errorfnc = VLPool(VL, self.GetPoolRun(), self.Data)
        if Errorfnc:
            VL.Exit(
                VLF.ErrorMessage(
                    "The following DPA routine(s) finished with errors:\n{}".format(
                        Errorfnc
                    )
                ),
                Cleanup=False,
            )

        VL.Logger(
            "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
            "~~~ DPA Analysis Complete ~~~\n"
            "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n",
            Print=True,
        )
        
