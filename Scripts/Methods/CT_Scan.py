import os
import sys

sys.dont_write_bytecode = True
from types import SimpleNamespace as Namespace

from Scripts.Common.VLParallel import VLPool
from Scripts.Common.utils import Method_base
from Scripts.Common.VLContainer import Container_Utils as Utils
from Scripts.VLPackages.GVXR.API import Run as CT_Scan, Dir as GVXRDir
from Scripts.VLPackages.ContainerInfo import GetInfo

class Method(Method_base):

    def Setup(self, VL, GVXRDicts, RunGVXR=True):
        def __init__(self, VL):
            super().__init__(VL)  # rune __init__ of Method_base
            self.MethodName = "GVXR"
            self.Containers_used = ["GVXR"]
        """
        GVXR - Simulation of X-ray CT scans
        """
        import json
        import glob
        if not (self.RunFlag and GVXRDicts):
            return
        # Call setup inside a GVXR container.
        funcname = "GVXR_Setup" # function to be executed within container
        funcfile = "{}/Setup.py".format(GVXRDir) # python file where 'funcname' is located
        PROJECT_DIR_CONT = Utils.host_to_container_path(VL.PROJECT_DIR)
        Setup = CT_Scan(funcfile, funcname, fnc_args=(GVXRDicts,PROJECT_DIR_CONT,VL.mode))
        Param_dir = f"{VL.PROJECT_DIR}/run_params/*.json"
        json_files = glob.glob(Param_dir)
        run_names = list(GVXRDicts.keys())
        for i,jsfile in enumerate(json_files):
            with open(jsfile) as f:
                self.Data[run_names[i]] = json.load(f)

    @staticmethod
    def PoolRun(VL,GVXRDict):
        funcname = "CT_scan" # function to be executed within container
        funcfile = "{}/CT_Scan.py".format(GVXRDir) # python file where 'funcname' is located
        RC = CT_Scan(funcfile, funcname, fnc_kwargs=GVXRDict)
        return RC
    

    def Run(self, VL):
        import Scripts.Common.VLFunctions as VLF
        if not self.Data:
            return
        VL.Logger("\n### Starting GVXR ###\n", Print=True)
        # for key in self.Data.keys():
        #     Errorfnc = self.PoolRun(VL,self.Data[key])
        #     if Errorfnc:
        #         VL.Exit(
        #             VLF.ErrorMessage(
        #                 "The following GVXR routine(s) finished with errors:\n{}".format(
        #                     Errorfnc
        #                 )
        #             )
        #         )
        Errorfnc = VLPool(VL, self.GetPoolRun(), self.Data)
        if Errorfnc:
            VL.Exit(
               VLF.ErrorMessage(
                   "\nThe following GVXR routine(s) finished with errors:\n{}".format(Errorfnc)
               ),
               Cleanup=False,
            )

        VL.Logger("\n### GVXR Complete ###", Print=True)