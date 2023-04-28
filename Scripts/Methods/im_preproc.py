import os
import sys

sys.dont_write_bytecode = True
from types import SimpleNamespace as Namespace

from Scripts.Common.VLParallel import VLPool
from Scripts.Common.utils import Method_base
from Scripts.Common.VLContainer import Container_Utils as Utils
# from Scripts.VLPackages.CIL.API import Run as CT_Recon, Dir as CilDIR
# from Scripts.VLPackages.CIL.Utils_IO import ReadNikonData, warn_Nikon
import VLconfig as VLC
import Scripts.Common.VLFunctions as VLF 

"""
module for performing preprocessing of images for use in Survos workflow.
Currently This is just normalization and registration.
"""

class Method(Method_base):
    def __init__(self, VL):
        super().__init__(VL)  # rune __init__ of Method_base
        self.MethodName = "ImPreProc"
        self.Containers_used = ["ImPreProc"]

    def Setup(self, VL, MethodDicts, Import=False):
        """Setup for Image Normalisation and registration with ITKElastix"""
        # if Run flag is False or MethodDicts is empty dont perform preprocessing and return instead.
        if not (self.RunFlag and MethodDicts):
            return
        self.Data = {}

        for MethodName, MethodParams in MethodDicts.items():
            Parameters = Namespace(**MethodParams)

            MethodDict = {
                "Name": MethodName,
            }
            # set location of Real Experimental data
            if hasattr(Parameters, "Exp_Data"):
                Methoddict["Exp_Data"] = Parameters.Exp_Data
            else:
                Methoddict["Exp_Data"] = "{}/Data".format(VL.PROJECT_DIR)
            # set location of simulated data
            if hasattr(Parameters, "Sim_Data"):
                Methoddict["Sim_Data"] = Parameters.Sim_Data
            else:
                Methoddict["Sim_Data"] = "{}/CIL-Images".format(VL.PROJECT_DIR)
            # set locvation of cad2vox data fro array mask
            if hasattr(Parameters, "Vox_Data"):
                Methoddict["Vox_Data"] = Parameters.Vox_Data
            else:
                Methoddict["Vox_Data"] = "{}/Voxel-Images".format(VL.PROJECT_DIR)

            ###################################################
            # Custom parameters for Image Registration
            ##################################################
            if hasattr(Parameters, "Reg_Params"):
                Methoddict["Reg_Params"] = Parameters.Reg_Params
            else:
                Methoddict["Reg_Params"] = None
            ###################################################
            # Normalization parameters
            ##################################################
            if hasattr(Parameters, "img_dim_x"):
                Methoddict['img_dim_x'] = Parameters.img_dim_x
            else:
                Methoddict['img_dim_x'] = 0

            if hasattr(Parameters, "img_dim_y"):
                Methoddict['img_dim_y'] = Parameters.img_dim_y
            else:
                Methoddict['img_dim_y'] = 0

            if hasattr(Parameters, "img_dim_z"):
                Methoddict['num_slices'] = Parameters.img_dim_z
            else:
                Methoddict['num_slices'] = 0
            
            if hasattr(Parameters,'raw_dtype'):
                Methoddict['raw_dtype'] = Parameters.raw_dtype
            else:
                Methoddict['raw_dtype'] = 'u2'

            if hasattr(Parameters,'nbins'):
                Methoddict['nbins'] = Parameters.nbins
            else:
                Methoddict['nbins'] = 512
            
            if hasattr(Parameters,'des_bg'):
                Methoddict['des_bg'] = Parameters.des_bg
            else:
                Methoddict['des_bg'] = 0.1

            if hasattr(Parameters,'des_fg'):
                Methoddict['des_fg'] = Parameters.des_fg
            else:
                Methoddict['des_fg'] = 0.9

            if hasattr(Parameters,'peak_width'):
                Methoddict['peak_width'] = Parameters.peak_width
            else:
                Methoddict['peak_width'] = 1

            if hasattr(Parameters,'set_order_no'):
                Methoddict['set_order_no'] = Parameters.set_order_no
            else:
                Methoddict['set_order_no'] = 10
            
            # catch any extra options and throw an error to say they are invalid
            param_dict = vars(Parameters)
            for key in ["Name","mesh"]:
                del param_dict[key]
            
            diff = set(param_dict.keys()).difference(MethodDict.keys())
            if list(diff) != []:
                invalid_options=''
                for i in list(diff):
                    invalid_options = invalid_options + f"PreProc.{i}={param_dict[i]}\n"
                VL.Exit(
                    VLF.ErrorMessage(
                        f"Invalid input parameters for Image Preprocessing:\n{invalid_options}"))
            self.Data[MethodName] = Methoddict.copy()
        return

    # *******************************************************************

    @staticmethod
    def PoolRun(VL,MethodDict,funcname):
        funcfile = "{}/Image_PreProc.py".format(MethodDIR) # python file where 'funcname' is located
        
        RC = ImPreProc(funcfile, funcname, fnc_kwargs=MethodDict)
        return RC
    

    def Run(self, VL, **kwargs):
        import Scripts.Common.VLFunctions as VLF
        if not self.Data:
            return
        VL.Logger("\n### Starting Image Pre-Processing ###\n", Print=True)
        # kwargs to control which processing steps occur
        Normalise = kwargs.get('Normalise',True)
        Register = kwargs.get('Register',True)
        # do Normalisation
        if Normalise:
            VL.Logger("\n### Performing Image Normalisation ###\n", Print=True)
            for key in self.Data.keys():
                Errorfnc = self.PoolRun(VL,self.Data[key],'Normalise')
                if Errorfnc:
                    VL.Exit(
                        VLF.ErrorMessage(
                            "The following Image Pre-Processing routine(s) finished with errors:\n{}".format(
                                Errorfnc
                            )
                        )
                    )
            VL.Logger("\n### Image Normalisation Complete ###\n", Print=True)
        # Do Registration
        if Register:
            VL.Logger("\n### Performing Image Registration ###\n", Print=True)
            for key in self.Data.keys():
                    Errorfnc = self.PoolRun(VL,self.Data[key],'Register')
                    if Errorfnc:
                        VL.Exit(
                            VLF.ErrorMessage(
                                "The following Image Pre-Processing routine(s) finished with errors:\n{}".format(
                                    Errorfnc
                                )
                            )
                        )
            VL.Logger("\n### Image Registration Complete ###\n", Print=True)
        VL.Logger("\n### Image Pre-Processing Complete ###", Print=True)        

        

