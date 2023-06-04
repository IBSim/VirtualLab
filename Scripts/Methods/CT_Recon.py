import os
import sys

sys.dont_write_bytecode = True
from types import SimpleNamespace as Namespace

from Scripts.Common.VLParallel import VLPool
from Scripts.Common.utils import Method_base
from Scripts.Common.VLContainer import Container_Utils as Utils
from Scripts.VLPackages.CIL.API import Run as CT_Recon, Dir as CilDIR
from Scripts.VLPackages.CIL.Utils_IO import ReadNikonData, warn_Nikon
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
                "output_dir":"{}/CIL_Images".format(VL.PROJECT_DIR),
                "Name": CILName,
            }
            # Define flag to display visualisations
            if VL.mode == "Headless":
                CILdict["Headless"] = True
            else:
                CILdict["Headless"] = False

            if hasattr(Parameters, "Nikon_file"):
                use_Nikon = True
                if os.path.isabs(Parameters.Nikon_file):
                #if abs path use that
                    if Utils.is_bound(Nikon_file,vlab_dir=VLC.VL_HOST_DIR):
                        Nikon_file = Parameters.Nikon_file
                    else:
                        message = "\n*************************************************************************\n" \
                                f"Error: The Nikon file '{Nikon_file}' is in a directory that'\n" \
                                "is not bound to the container. This can be corrected either using the \n" \
                                "--bind option or by including the argument bind in VLconfig.py\n" \
                                "*************************************************************************\n"
                        raise FileNotFoundError(message) 
                else:
                # if not abs path check the input directory
                    Nikon_file = f'{VL.PARAMETERS_DIR}/{Parameters.Nikon_file}'
                
                if  os.path.exists(Nikon_file):
                    print(f"Reading CIL parameters from Nikon file: {Nikon_file}")
                else:
                    # convert file path from container to host if necessary so errors make sense
                    Nikon_file=Utils.container_to_host_path(Nikon_file)
                    raise FileNotFoundError(f"Could not find Nikon file {Nikon_file}\n \
                    Please check the file is in the input directory {VLC.VL_HOST_DIR}/{VL.Project}/{VL.Simulation} \n \
                    or that path to this file is correct.")
                CILdict = ReadNikonData(CILdict,Nikon_file)
            else:
                use_Nikon = False


            if hasattr(Parameters, "Beam_PosX") and hasattr(Parameters, "Beam_PosY") and hasattr(Parameters, "Beam_PosZ"):
                warn_Nikon(use_Nikon,"Beam_Pos")
                CILdict["Beam"] = [
                    Parameters.Beam_PosX,
                    Parameters.Beam_PosY,
                    Parameters.Beam_PosZ,
                ]

            if hasattr(Parameters, "Spacing_X"):
                warn_Nikon(use_Nikon,"Spacing_X")
                CILdict["Spacing_X"] = Parameters.Spacing_X

            if hasattr(Parameters, "Spacing_Y"):
                warn_Nikon(use_Nikon,"Spacing_Y")
                CILdict["Spacing_Y"] = Parameters.Spacing_Y

            if hasattr(Parameters, "Detect_PosX") and hasattr(Parameters, "Detect_PosY") and hasattr(Parameters, "Detect_PosZ"):
                warn_Nikon(use_Nikon,"Detect_Pos")
                CILdict["Detector"] = [
                    Parameters.Detect_PosX,
                    Parameters.Detect_PosY,
                    Parameters.Detect_PosZ,
                ]

            if hasattr(Parameters, "Model_PosX") and hasattr(Parameters, "Model_PosY") and hasattr(Parameters, "Model_PosZ"):
                CILdict["Model"] = [
                    Parameters.Model_PosX,
                    Parameters.Model_PosY,
                    Parameters.Model_PosZ,
                ]
            else:
                CILdict["Model"] = [0,0,0]

            if hasattr(Parameters, "Pix_X"):
                warn_Nikon(use_Nikon,"Pix_X")
                CILdict["Pix_X"] = Parameters.Pix_X

            if hasattr(Parameters, "Pix_Y"):
                warn_Nikon(use_Nikon,"Pix_X")
                CILdict["Pix_Y"] = Parameters.Pix_Y

            if hasattr(Parameters, "num_projections"):
                warn_Nikon(use_Nikon,"num_projections")
                CILdict["num_projections"] = Parameters.num_projections

            if hasattr(Parameters, "angular_step"):
                warn_Nikon(use_Nikon,"angular_step")
                CILdict["angular_step"] = Parameters.angular_step

            if hasattr(Parameters, "image_format"):
                CILdict["im_format"] = Parameters.image_format

            if hasattr(Parameters, "bitrate"):
                CILdict["bitrate"] = Parameters.bitrate
                
            if hasattr(Parameters, "Recon_Method"):
                if Parameters.Recon_Method.upper() not in ['FBP','FDK']:
                    raise ValueError(f"Invalid Recon_Method must be one of FBP or FDK")
                CILdict["backend"] = Parameters.Recon_Method.upper()
            if hasattr(Parameters, "Beam_Type"):
                if Parameters.Recon_Method.upper() not in ['point','parallel']:
                    raise ValueError(f"Invalid Beam_Type must be one of point or parallel")
                CILdict["Beam_Type"] = Parameters.Beam_Type
            
            self.Data[CILName] = CILdict.copy()
        return

    # *******************************************************************

    @staticmethod
    def PoolRun(VL,CilDict,funcname):
        funcfile = "{}/CT_reconstruction.py".format(CilDIR) # python file where 'funcname' is located
        
        RC = CT_Recon(funcfile, funcname, fnc_kwargs=CilDict)
        return RC
    

    def Run(self, VL, **kwargs):
        import Scripts.Common.VLFunctions as VLF
        if not self.Data:
            return
        VL.Logger("\n### Starting CT Reconstruction ###\n", Print=True)
        for key in self.Data.keys():
            Errorfnc = self.PoolRun(VL,self.Data[key],'CT_Recon')
            if Errorfnc:
                VL.Exit(
                    VLF.ErrorMessage(
                        "The following CIL routine(s) finished with errors:\n{}".format(
                            Errorfnc
                        )
                    )
                )
        if kwargs.get('Helical',False):
            # assemble slices from a helical scan into a single tiff file.
            VL.Logger("\n### Assembling slices from Helical scan ###\n", Print=True)
                        
            for key in self.Data.keys():
                Errorfnc = self.PoolRun(VL,self.Data[key],'Helix')
                if Errorfnc:
                    VL.Exit(
                        VLF.ErrorMessage(
                            "The following Image CIL routine(s) finished with errors:\n{}".format(
                                Errorfnc
                            )
                        )
                    )
            VL.Logger("\n### Slice Assembly Complete ###\n", Print=True)

        VL.Logger("\n### CT Reconstruction Complete ###", Print=True)        

        

