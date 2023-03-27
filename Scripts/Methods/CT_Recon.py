import os
import sys

sys.dont_write_bytecode = True
from types import SimpleNamespace as Namespace

from Scripts.Common.VLParallel import VLPool
from Scripts.Common.utils import Method_base
from Scripts.Common.VLContainer import Container_Utils as Utils
from Scripts.VLPackages.CIL.API import Run as CT_Recon, Dir as CilDIR
import VLconfig as VLC

"""
Template file for creating a new method. Create a copy of this file as #MethodName.py,
which is the name of the new method, and edit as desired. Any file starting with
_ are ignored.
"""


class Method(Method_base):
    def __init__(self, VL):
        super().__init__(VL)  # rune __init__ of Method_base
        self.MethodName = "CIL"
        self.Containers_used = ["CIL"]

    def Setup(self, VL, CILdicts, Import=False):
        """setup CT reconstruction with CIL"""
        # if RunCIL is False or CILdicts is empty dont perform Simulation and return instead.
        if not (self.RunFlag and CILdicts):
            return
        self.Data = {}

        for CILName, CILParams in CILdicts.items():
            Parameters = Namespace(**CILParams)

            CILdict = {
                "work_dir": "{}/GVXR-Images".format(VL.PROJECT_DIR),
                "Name": CILName,
            }
            # Define flag to display visualisations
            if VL.mode == "Headless":
                CILdict["Headless"] = True
            else:
                CILdict["Headless"] = False

            if hasattr(Parameters, "Nikon_file"):
                if os.path.isabs(Parameters.Nikon_file):
                #if abs path use that 
                # Note: we need to now convert it to a path is accessible within the container
                    Nikon_file = Utils.host_to_container_path(Parameters.Nikon_file)
                    Nikon_file = Parameters.Nikon_file
                else:
                # if not abs path check the input directory
                    Nikon_file = f'{VL.PARAMETERS_DIR}/{Parameters.Nikon_file}'
                
                if os.path.exists(Nikon_file):
                    print(f"Reading CIL parameters from Nikon file: {Nikon_file}")
                else:
                    # convert file path from container to host if necessary so errors make sense
                    Nikon_file=Utils.container_to_host_path(Nikon_file)
                    raise FileNotFoundError(f"Could not find Nikon file {Nikon_file}\n \
                    Please check the file is in the input directory {VLC.VL_HOST_DIR}/{VL.Project}/{VL.Simulation} \n \
                    or that path to this file is correct.")
                CILdict["Nikon"] = Nikon_file
            else:
                CILdict["Nikon"] = None

            # if hasattr(Parameters,'Beam_Pos_units'):
            #    CILdict['Beam_Pos_units'] = Parameters.Beam_Pos_units
            # else:
            #    CILdict['Beam_Pos_units'] = 'm'

            if hasattr(Parameters, "Beam_PosX") and hasattr(Parameters, "Beam_PosY") and hasattr(Parameters, "Beam_PosZ"):
                CILdict["Beam"] = [
                    Parameters.Beam_PosX,
                    Parameters.Beam_PosY,
                    Parameters.Beam_PosZ,
                ]
            else:
                CILdict["Beam"] = [0,0,0]

            # if hasattr(Parameters,'Detect_Pos_units'):
            #    CILdict['Det_Pos_units'] = Parameters.Detect_Pos_units
            # else:
            #    CILdict['Det_Pos_units'] = 'm'

            if hasattr(Parameters, "Spacing_X"):
                CILdict["Spacing_X"] = Parameters.Spacing_X
            else:
                CILdict["Spacing_X"] = 0.5

            if hasattr(Parameters, "Spacing_Y"):
                CILdict["Spacing_Y"] = Parameters.Spacing_Y
            else:
                CILdict["Spacing_Y"] = 0.5
            if hasattr(Parameters, "Detect_PosX") and hasattr(Parameters, "Detect_PosY") and hasattr(Parameters, "Detect_PosZ"):
                CILdict["Detector"] = [
                    Parameters.Detect_PosX,
                    Parameters.Detect_PosY,
                    Parameters.Detect_PosZ,
                ]
            else:
                CILdict["Detector"] = [0,0,0]
            if hasattr(Parameters, "Model_PosX") and hasattr(Parameters, "Model_PosY") and hasattr(Parameters, "Model_PosZ"):
                CILdict["Model"] = [
                    Parameters.Model_PosX,
                    Parameters.Model_PosY,
                    Parameters.Model_PosZ,
                ]
            else:
                CILdict["Model"] = [0,0,0]

            if hasattr(Parameters, "Pix_X"):
                CILdict["Pix_X"] = Parameters.Pix_X
            else:
                CILdict["Pix_X"] = 1

            if hasattr(Parameters, "Pix_Y"):
                CILdict["Pix_Y"] = Parameters.Pix_Y
            else:
                CILdict["Pix_Y"] = 1
            # if hasattr(Parameters,'Model_Pos_units'):
            #    CILdict['Model_Pos_units'] = Parameters.Model_Pos_units
            # else:
            #    CILdict['Model_Pos_units'] = 'm'

            # if hasattr(Parameters, "rotation"):
            #     CILdict["rotation"] = Parameters.rotation

            if hasattr(Parameters, "num_projections"):
                CILdict["num_projections"] = Parameters.num_projections

            if hasattr(Parameters, "angular_step"):
                CILdict["angular_step"] = Parameters.angular_step

            if hasattr(Parameters, "image_format"):
                CILdict["im_format"] = Parameters.image_format

            self.Data[CILName] = CILdict.copy()
        return

    # *******************************************************************

    @staticmethod
    def PoolRun(VL,CilDict):
        funcname = "CT_Recon" # function to be executed within container
        funcfile = "{}/CT_reconstruction.py".format(CilDIR) # python file where 'funcname' is located
        
        RC = CT_Recon(funcfile, funcname, fnc_kwargs=CilDict)
        return RC
    

    def Run(self, VL, **kwargs):
        import Scripts.Common.VLFunctions as VLF
        if not self.Data:
            return
        VL.Logger("\n### Starting CT Reconstruction ###\n", Print=True)
        for key in self.Data.keys():
            Errorfnc = self.PoolRun(VL,self.Data[key])
            if Errorfnc:
                VL.Exit(
                    VLF.ErrorMessage(
                        "The following CIL routine(s) finished with errors:\n{}".format(
                            Errorfnc
                        )
                    )
                )

        VL.Logger("\n### CT Reconstruction Complete ###", Print=True)        

        

