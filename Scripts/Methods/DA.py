import os
import sys

sys.dont_write_bytecode = True
from types import SimpleNamespace as Namespace

import Scripts.Common.VLFunctions as VLF
from Scripts.Common.VLParallel import VLPool
from Scripts.Common.utils import Method_base
from Scripts.Common.VLContainer import Container_Utils as Utils

"""
DA - Data Analysis
"""


class Method(Method_base):
    def __init__(self, VL):
        super().__init__(VL)  # run __init__ of Method_base
        self.MethodName = "DA"

    def Setup(self, VL, DADicts, Import=False):
        # if either DADicts is empty or RunDA is False we will return
        if not (self.RunFlag and DADicts):
            return

        VL.tmpDA_DIR = "{}/DA".format(VL.TEMP_DIR)
        os.makedirs(VL.tmpDA_DIR, exist_ok=True)

        FileDict = {}
        for DAName, ParaDict in DADicts.items():
            CALC_DIR = "{}/{}".format(VL.PROJECT_DIR, DAName)
            if Import:
                ParaDict = VLF.ImportUpdate(
                    "{}/Parameters.py".format(CALC_DIR), ParaDict
                )

            # ======================================================================
            # get file path & perform checks
            # default name is Single
            file_name, func_name = VLF.FileFuncSplit(ParaDict["File"], "Single")

            if (file_name, func_name) not in FileDict:
                # Check file in directory & get path
                FilePath = VLF.GetFilePath(
                    [VL.SIM_DA, VL.VLRoutine_SCRIPTS],
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
            DADict = {
                "CALC_DIR": CALC_DIR,
                "TMP_CALC_DIR": "{}/{}".format(VL.tmpDA_DIR, DAName),
                "Parameters": Namespace(**ParaDict),
                "FileInfo": File_func,
                "Data": {},
            }

            # Important information can be added to Data during any stage of the
            # data analysis, and this will be saved to the location specified by the
            # value for the __file__ key
            DADict["Data"] = {"__file__": "{}/Data.pkl".format(DADict["CALC_DIR"])}

            if VL.mode in ("Headless", "Continuous"):
                DADict["LogFile"] = "{}/Output.log".format(DADict["CALC_DIR"])
            else:
                DADict["LogFile"] = None

            os.makedirs(DADict["TMP_CALC_DIR"], exist_ok=True)

            self.Data[DAName] = DADict

    @staticmethod
    def PoolRun(VL, DADict):

        Parameters = DADict["Parameters"]

        os.makedirs(DADict["CALC_DIR"], exist_ok=True)
        VLF.WriteData("{}/Parameters.py".format(DADict["CALC_DIR"]), Parameters)

        DASgl = VLF.GetFunc(*DADict["FileInfo"])

        err = DASgl(VL, DADict)

        return err

    def Run(self, VL):
        VL.AddToPath(VL.SIM_DA,0)

        VL.Logger(
            "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
            "~~~ Starting Data Analysis ~~~\n"
            "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n",
            Print=True,
        )

        Errorfnc = VLPool(VL, self.GetPoolRun(), self.Data)
        if Errorfnc:
            VL.Exit(
                VLF.ErrorMessage(
                    "The following DA routine(s) finished with errors:\n{}".format(
                        Errorfnc
                    )
                ),
                Cleanup=False,
            )

        VL.Logger(
            "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
            "~~~ Data Analysis Complete ~~~\n"
            "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n",
            Print=True,
        )
