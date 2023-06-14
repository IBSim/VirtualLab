import os
import sys

sys.dont_write_bytecode = True
from types import SimpleNamespace as Namespace

from Scripts.Common.VLParallel import VLPool
from Scripts.Common.utils import Method_base
from Scripts.Common.VLContainer import Container_Utils as Utils
from Scripts.VLPackages.ImPreProc.API import Run as ImPreProc, Dir as MethodDIR
import VLconfig as VLC
import Scripts.Common.VLFunctions as VLF
from pathlib import Path

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
                Exp_Data = self.check_data(VL,VLF,Parameters.Exp_Data,category="EXP")
                MethodDict["Exp_Data"] = Exp_Data
            else:
                VL.Exit(
                    VLF.ErrorMessage(
                        f"You must provide parameter ImPreProc.EXP_Data to give location of experimental data to process."))
            # set location of simulated data
            if hasattr(Parameters, "Sim_Data"):
                Sim_Data = self.check_data(VL,VLF,Parameters.Sim_Data,category="SIM")
                MethodDict["Sim_Data"] = Sim_Data
            else:
                MethodDict["Sim_Data"] = None
            # set location of cad2vox data for use as an array mask
            if hasattr(Parameters, "Vox_Data"):
                Vox_data = check_data(VL,VLF,Parameters.Vox_Data,category="VOX")
                MethodDict["Vox_Data"] = Vox_Data
            else:
                MethodDict["Vox_Data"] = None

            ###################################################
            # Custom parameters for Image Registration
            ##################################################
            if hasattr(Parameters, "Reg_Params"):
                if isinstance(Parameters.Reg_Params, str):
                    MethodDict["Reg_Params"] = [Parameters.Reg_Params]
                elif isinstance(Parameters.Reg_Params, list):
                    MethodDict["Reg_Params"] = Parameters.Reg_Params
                else:
                    VL.Exit(VLF.ErrorMessage(f"preproc.Reg_Params \
                        must be a string or list of parameter files."))

            if hasattr(Parameters, "Iterations"):
                if isinstance(Parameters.Iterations, int) and Parameters.Iterations >0:
                    MethodDict["Iterations"] = Parameters.Iterations
                else:
                    VL.Exit(VLF.ErrorMessage(f"Invalid value {Parameters.Iterations} \
                    for preproc.Iterations, must be an integer greater than 0."))

            if hasattr(Parameters, "Samples"):
                if isinstance(Parameters.Samples, int) and Parameters.Samples > 0:
                    MethodDict["Samples"] = Parameters.Samples
                else:
                    VL.Exit(VLF.ErrorMessage(f"Invalid value {Parameters.Samples}  \
                        for preproc.Samples, must be an integer greater than 0."))

            if hasattr(Parameters, "Resolutions"):
                if isinstance(Parameters.Resolutions, int) and Parameters.Resolutions > 0:
                    MethodDict["Resolutions"] = Parameters.Resolutions   
                else:
                    VL.Exit(VLF.ErrorMessage(f"Invalid value {Parameters.Resolutions}  \
                        for preproc.Resolutions, must be an integer greater than 0."))

            ###################################################
            # Normalization parameters
            ##################################################
            if hasattr(Parameters, "img_dim_x"):
                MethodDict['img_dim_x'] = Parameters.img_dim_x
            else:
                MethodDict['img_dim_x'] = 0

            if hasattr(Parameters, "img_dim_y"):
                MethodDict['img_dim_y'] = Parameters.img_dim_y
            else:
                MethodDict['img_dim_y'] = 0

            if hasattr(Parameters, "num_slices"):
                MethodDict['num_slices'] = Parameters.num_slices
            else:
                MethodDict['num_slices'] = 0
            
            if hasattr(Parameters,'raw_dtype'):
                MethodDict['raw_dtype'] = Parameters.raw_dtype
            else:
                MethodDict['raw_dtype'] = 'u2'

            if hasattr(Parameters,'nbins'):
                MethodDict['nbins'] = Parameters.nbins
            else:
                MethodDict['nbins'] = 512
            
            if hasattr(Parameters,'des_bg'):
                MethodDict['des_bg'] = Parameters.des_bg
            else:
                MethodDict['des_bg'] = 0.1

            if hasattr(Parameters,'des_fg'):
                MethodDict['des_fg'] = Parameters.des_fg
            else:
                MethodDict['des_fg'] = 0.9

            if hasattr(Parameters,'peak_width'):
                MethodDict['peak_width'] = Parameters.peak_width
            else:
                MethodDict['peak_width'] = 1

            if hasattr(Parameters,'set_order_no'):
                MethodDict['set_order_no'] = Parameters.set_order_no
            else:
                MethodDict['set_order_no'] = 10
                
            if hasattr(Parameters,'Keep_Raw'):
                MethodDict['Keep_Raw'] = Parameters.Keep_Raw
            else:
                MethodDict['Keep_Raw'] = False

            # catch any extra options and throw an error to say they are invalid
            param_dict = vars(Parameters)
            for key in ["Name"]:
                del param_dict[key]
            
            diff = set(param_dict.keys()).difference(MethodDict.keys())
            if list(diff) != []:
                invalid_options=''
                for i in list(diff):
                    invalid_options = invalid_options + f"PreProc.{i}={param_dict[i]}\n"
                VL.Exit(
                    VLF.ErrorMessage(
                        f"Invalid input parameters for Image Preprocessing:\n{invalid_options}"))
            self.Data[MethodName] = MethodDict.copy()
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
                        if Errorfnc == 137:
                            VL.Exit(
                                VLF.ErrorMessage(
                                    "Image Pre-Processing routine(s) was killed as the system ran out of avalible memory\n"
                                    )
                                )
                        else:
                            VL.Exit(
                                VLF.ErrorMessage(
                                    "Image Pre-Processing routine(s) finished with errors:\n Exit code:{}".format(
                                        Errorfnc
                                    )
                                )
                        )
            VL.Logger("\n### Image Registration Complete ###\n", Print=True)
        VL.Logger("\n### Image Pre-Processing Complete ###", Print=True)        

    @staticmethod 
    def check_data(VL,VLF,filename,category="EXP"):
        P = Path(filename)
        # check if filename is absolute path to a file that exists in which case return
        if P.is_absolute() and P.is_file():
            return filename
        # check data category to change the default directory in which to look
        if category == "EXP":
            default_dir = Path(f"{VL.PARAMETERS_DIR}/Data/{filename}")
        elif category == "SIM":
            if P.is_dir():
                return filename
            default_dir = Path(f"{VL.PROJECT_DIR}/CIL_Images/{filename}")
        elif category == "VOX":
            default_dir = Path(f"{VL.Project_DIR}/Voxel-Images/{filename}")
        else:
            VL.Exit(VLF.ErrorMessage(f"Unknown data category {category} must be one of VOX, SIM or EXP."))
        # path is not absolute so check file is in default location and if not throw error
        if default_dir.is_file():
            return str(default_dir)
        else:
            VL.Exit(VLF.ErrorMessage(f"Invalid parameter ImPreProc.{cateagory}_Data the file {default_dir} does not exist."))
