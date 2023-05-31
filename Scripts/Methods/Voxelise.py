import os
import sys

sys.dont_write_bytecode = True
from types import SimpleNamespace as Namespace
from importlib import import_module
import copy

import Scripts.Common.VLFunctions as VLF
from Scripts.Common.VLParallel import VLPool
from Scripts.Common.utils import Method_base
from Scripts.Common.VLContainer import Container_Utils as Utils
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
            for key in ["Name", "mesh"]:
                del param_dict[key]

            if hasattr(Parameters, "Nikon_file"):
                if os.path.isabs(Parameters.Nikon_file):
                    # if abs path use that
                    if Utils.is_bound(Nikon_file, vlab_dir=VLC.VL_HOST_DIR):
                        Nikon_file = Parameters.Nikon_file
                    else:
                        message = (
                            "\n*************************************************************************\n"
                            f"Error: The Nikon file '{Nikon_file}' is in a directory that'\n"
                            "is not bound to the container. This can be corrected either using the \n"
                            "--bind option or by including the argument bind in VLconfig.py\n"
                            "*************************************************************************\n"
                        )
                        raise FileNotFoundError(message)
                else:
                    # if not abs path check the input directory
                    Nikon_file = f"{VL.PARAMETERS_DIR}/{Parameters.Nikon_file}"

                if os.path.exists(Nikon_file):
                    print(f"Reading Voxel Unit Length from Nikon file: {Nikon_file}")
                else:
                    # convert file path from container to host if necessary so errors make sense
                    Nikon_file = Utils.container_to_host_path(Nikon_file)
                    raise FileNotFoundError(
                        f"Could not find Nikon file {Nikon_file}\n \
                    Please check the file is in the input directory {VLC.VL_HOST_DIR}/{VL.Project}/{VL.Simulation} \n \
                    or that path to this file is correct."
                    )

                Voxdict["unit_length"] = ReadNikonData(Nikon_file)

            diff = set(param_dict.keys()).difference(VoxDict.keys())
            if list(diff) != []:
                invalid_options = ""
                for i in list(diff):
                    invalid_options = invalid_options + f"Vox.{i}={param_dict[i]}\n"
                VL.Exit(
                    VLF.ErrorMessage(
                        f"Invalid input parameters for cad2vox:\n{invalid_options}"
                    )
                )

            self.Data[VoxName] = VoxDict.copy()
        return

    @staticmethod
    def PoolRun(VL, VoxDict):
        funcname = "voxelise"  # function to be executed within container
        funcfile = "{}/main.py".format(
            VoxDir
        )  # python file where 'funcname' is located

        RC = cad2vox(funcfile, funcname, fnc_kwargs=VoxDict)
        return RC

    @staticmethod
    def ReadNikonData(Nikon_file):
        """
        Function to read in Nikon xtect file and extract the Voxel unit length in x,y and z.

        Param: Nikon_file - str - path to .xtect input file.

        Return: list of floats corresponding to unit length in x,y and z
        """

        # parse xtek file
        with open(Nikon_file, "r") as f:
            content = f.readlines()

        content = [x.strip() for x in content]

        unit_length = [None, None, None]
        for line in content:
            # number of projections
            if line.startswith("VoxelSizeX"):
                unit_length[0] = float(line.split("=")[1])
            if line.startswith("VoxelSizeY"):
                unit_length[1] = float(line.split("=")[1])
            if line.startswith("VoxelSizeZ"):
                unit_length[2] = float(line.split("=")[1])

        if None in unit_length:
            raise ValueError(
                f"The provided nikon file does not appear to correctly define the unit length. \
                Please check the file defines the VoxelSize for x, y and z. Unit_length has \
                been read as: {unit_length}")
        
        return unit_length

    def Run(self, VL):
        if not self.Data:
            return
        VL.Logger("\n### Starting Voxelisation ###\n", Print=True)

        Errorfnc = VLPool(VL, self.GetPoolRun(), self.Data)
        if Errorfnc:
            VL.Exit(
                VLF.ErrorMessage(
                    "\nThe following Cad2Vox routine(s) finished with errors:\n{}".format(
                        Errorfnc
                    )
                ),
                Cleanup=False,
            )

        VL.Logger("\n### Voxelisation Complete ###", Print=True)
