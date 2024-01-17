import os
import sys

sys.dont_write_bytecode = True
from types import SimpleNamespace as Namespace

import Scripts.Common.VLFunctions as VLF
from Scripts.Common.VLParallel import VLPool
from Scripts.Common.utils import Method_base
from Scripts.Common.VLContainer import Container_Utils as Utils
from Scripts.VLPackages.modelib.API import Run as modelib
"""
Modelib - dislocation dynamics analysis
"""


class Method(Method_base):
    def __init__(self, VL):
        super().__init__(VL)  # run __init__ of Method_base
        self.MethodName = "modelib"
        self.Containers_used = ["modelib"]
    def Setup(self, VL, modelibDicts, Import=False):
        # if either modelibDicts is empty or Runmodelib is False we will return
        if not (self.RunFlag and modelibDicts):
            return

        VL.tmpmodelib_DIR = "{}/modelib".format(VL.TEMP_DIR)
        os.makedirs(VL.tmpmodelib_DIR, exist_ok=True)

        FileDict = {}
        for modelibName, ParaDict in modelibDicts.items():
            Parameters = Namespace(**ParaDict)
            CALC_DIR = "{}/{}".format(VL.PROJECT_DIR, modelibName)
            modelibDict = {'Name':modelibName,
                 'CALC_DIR':CALC_DIR,
                 'PREdd1':"{}".format(CALC_DIR),
                 'PREdd':"{}/inputFiles".format(CALC_DIR),
	         'PREdd2':"{}/evl".format(CALC_DIR),
		 'PREdd4':"{}/E".format(CALC_DIR),
                 'PREdd3':"{}/F".format(CALC_DIR),
                 'Add':"{}/Add".format(CALC_DIR),
                 'TMP_CALC_DIR':"{}/{}".format(VL.tmpmodelib_DIR, modelibName),
                 'Parameters':Namespace(**ParaDict),
                 'Data':{}}
            if VL.mode in ("Headless", "Continuous"):
                modelibDict["LogFile"] = "{}/Output.log".format(modelibDict["CALC_DIR"])
            else:
                modelibDict["LogFile"] = None

            os.makedirs(modelibDict["TMP_CALC_DIR"], exist_ok=True)

            
          
         

            # ==========================================================================
            # Create dictionary for each analysis
            
            # Important information can be added to Data during any stage of the
            # data analysis, and this will be saved to the location specified by the
            # value for the __file__ key
         
           
            self.Data[modelibName] = modelibDict

    @staticmethod
    def PoolRun(VL, modelibDict):

       

        os.makedirs(modelibDict["CALC_DIR"], exist_ok=True)
       
        os.makedirs(modelibDict['PREdd'],exist_ok=True)
        os.makedirs(modelibDict['PREdd2'],exist_ok=True)
        os.makedirs(modelibDict['PREdd3'],exist_ok=True)
        os.makedirs(modelibDict['PREdd4'],exist_ok=True)
        funcname = "microstructure" # function to be executed within container
        funcfile = "{}/DDD.py".format(VL.SIM_MODELIB) # python file where 'funcname' is located
        
        RC = modelib(funcfile, funcname, fnc_kwargs=modelibDict)
        return RC

    def Run(self, VL):
        VL.AddToPath(VL.SIM_MODELIB,0)

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
                    "The following modelib routine(s) finished with errors:\n{}".format(
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
