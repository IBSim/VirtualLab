import os
import sys

sys.dont_write_bytecode = True
from types import SimpleNamespace as Namespace
from importlib import import_module
import copy

import Scripts.Common.VLFunctions as VLF          
from Scripts.Common.VLParallel import VLPool
from Scripts.Common.utils import Method_base
from Scripts.VLPackages.Vox.API import Run as cad2vox, Dir as VoxDir


def Check_Threads(num_threads):
    """Function to check the user defined Number of OpenMP threads are valid"""
    try:
        int(num_threads)
    except ValueError:
        print(num_threads)
        raise ValueError(
            "Invalid number of threads for Cad2Vox, must be an Integer value, "
            "or castable to and Integer value"
        )

    if int(num_threads) < 0:
        raise ValueError(
            "Invalid Number of threads for Cad2Vox. Must be greater than 0"
        )


class Method(Method_base):
    def __init__(self, VL):
        super().__init__(VL)  # rune __init__ of Method_base
        self.MethodName = "Vox"
        self.Containers_used = ["Cad2Vox"]
    def Setup(self, VL, VoxDicts, Import=False):
        """
        Vox - Mesh Voxelisation using Cuda or OpenMP
        """
        VL.OUT_DIR = "{}/Voxel-Images".format(VL.PROJECT_DIR)
        VL.MESH_DIR = "{}/Meshes".format(VL.PROJECT_DIR)

        # if RunVoxelise is False or VoxDicts is empty dont perform voxelisation and return instead.
        if not (self.RunFlag and VoxDicts):
            return

        if not os.path.exists(VL.OUT_DIR):
            os.makedirs(VL.OUT_DIR)

        for VoxName, VoxParams in VoxDicts.items():
            Parameters = Namespace(**VoxParams)
            # check mesh for file extension and if not present assume salome med
            root, ext = os.path.splitext(Parameters.mesh)
            if not ext:
                ext = ".med"
            mesh = root + ext
            # If mesh is an absolute path use it
            if os.path.isabs(mesh):
                IN_FILE = mesh
            # If not assume the file is in the Mesh directory
            else:
                IN_FILE = "{}/{}".format(VL.MESH_DIR, mesh)
            VoxDict = {}
            VoxDict["input_file"] = IN_FILE
            VoxDict["output_file"] = "{}/{}".format(VL.OUT_DIR, VoxName)

            # handle optional arguments
            if hasattr(Parameters, "unit_length"):
                VoxDict["unit_length"] = Parameters.unit_length

            if hasattr(Parameters, "gridsize"):
                VoxDict["gridsize"] = Parameters.gridsize
            # Logic to handle placing greyscale file in the correct place. i.e. in the output dir not the run directory.
            if hasattr(Parameters, "greyscale_file") and os.path.isabs(
                Parameters.greyscale_file
            ):
                # Abs. paths go where they say
                VoxDict["greyscale_file"] = Parameters.greyscale_file
            elif hasattr(Parameters, "greyscale_file") and not os.path.isabs(
                Parameters.greyscale_file
            ):
                # This makes a non abs. path relative to the output directory not the run directory (for consistency)
                VoxDict["greyscale_file"] = "{}/{}".format(
                    VL.OUT_DIR, Parameters.greyscale_file
                )
            else:
                # greyscale not given so generate a file in the output directory
                VoxDict["greyscale_file"] = "{}/greyscale_{}.csv".format(
                    VL.OUT_DIR, VoxName
                )

            if hasattr(Parameters, "use_tetra"):
                VoxDict["use_tetra"] = Parameters.use_tetra

            if hasattr(Parameters, "cpu"):
                VoxDict["cpu"] = Parameters.cpu

            if hasattr(Parameters, "solid"):
                VoxDict["solid"] = Parameters.solid

            if hasattr(Parameters, "Num_Threads"):
                Check_Threads(Parameters.Num_Threads)
                VoxDict["Num_Threads"] = Parameters.Num_Threads

            if hasattr(Parameters, "image_format"):
                VoxDict["im_format"] = Parameters.image_format

            if hasattr(Parameters, "Orientation"):
                VoxDict["Orientation"] = Parameters.Orientation
                
            if hasattr(Parameters, "Output_Resolution"):
                VoxDict["Output_Resolution"] = Parameters.Output_Resolution
                 
            # catch any extra options and throw an error to say they are invalid
            param_dict = vars(Parameters)
            for key in ["Name","mesh"]:
                del param_dict[key]
            
            diff = set(param_dict.keys()).difference(VoxDict.keys())
            if list(diff) != []:
                invalid_options=''
                for i in list(diff):
                    invalid_options = invalid_options + f"Vox.{i}={param_dict[i]}\n"
                VL.Exit(
                    VLF.ErrorMessage(
                        f"Invalid input parameters for cad2vox:\n{invalid_options}"))
            
            self.Data[VoxName] = VoxDict.copy()
        return

    @staticmethod
    def PoolRun(VL,VoxDict):
        funcname = "voxelise" # function to be executed within container
        funcfile = "{}/main.py".format(VoxDir) # python file where 'funcname' is located
        
        RC = cad2vox(funcfile, funcname, fnc_kwargs=VoxDict)
        return RC
    


    def Run(self, VL):
        if not self.Data:
            return
        VL.Logger("\n### Starting Voxelisation ###\n", Print=True)

        Errorfnc = VLPool(VL, self.GetPoolRun(), self.Data)
        if Errorfnc:
            VL.Exit(
               VLF.ErrorMessage(
                   "\nThe following Cad2Vox routine(s) finished with errors:\n{}".format(Errorfnc)
               ),
               Cleanup=False,
            )

        VL.Logger("\n### Voxelisation Complete ###", Print=True)


   


    
    
    
    
