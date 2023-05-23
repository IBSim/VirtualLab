def GVXR_Setup(GVXRDicts, PROJECT_DIR,PARAMETERS_DIR, mode):
    """
    GVXR - Simulation of X-ray CT scans
    """
    from importlib import import_module
    import copy
    import pydantic
    import pickle
    import os
    import sys
    from types import SimpleNamespace as Namespace
    from pydantic.dataclasses import dataclass, Field
    from typing import Optional, List
    from Scripts.Common.VLContainer.Container_Utils import (
        container_to_host_path,
    )
    from Scripts.VLPackages.GVXR.Utils_IO import ReadNikonData
    from Scripts.VLPackages.GVXR.GVXR_utils import (
        Check_Materials,
        dump_to_json,
        warning_message,
    )
    import VLconfig as VLC

    # list of all accepted params for GVXR
    GVXR_parameters = [
        "Nikon_file",
        "Material_list",
        "Amounts",
        "Density",
        "energy_units",
        "Tube_Angle",
        "Tube_Voltage",
        "Energy",
        "Intensity",
        "num_projections",
        "angular_step",
        "image_format",
        "use_tetra",
        "downscale",
        "Beam_PosX",
        "Beam_PosY",
        "Beam_PosZ",
        "Beam_Pos_units",
        "Beam_Type",
        "Detect_PosX",
        "Detect_PosY",
        "Detect_PosZ",
        "Detect_Pos_units",
        "Pix_X",
        "Pix_Y",
        "SpacingX",
        "SpacingY",
        "Spacing_units",
        "Model_PosX",
        "Model_PosY",
        "Model_PosZ",
        "Model_Pos_units",
        "Model_Mesh_units",
        "rotation",
        "fill_percent",
    ]

    def convert_tets_to_tri(mesh_file):
        """
        Function to read in a tetrahedron based
        volume mesh with meshio and convert it
        into surface triangle mesh for use with
        GVXR.
        """
        import numpy as np
        import meshio
        from Scripts.VLPackages.GVXR.GVXR_utils import tets2tri, find_the_key
        import os

        root, ext = os.path.splitext(mesh_file)
        new_mesh_file = f"{root}_triangles{ext}"
        # This check helps us avoid having to repeat the conversion from tri to tet
        # when using one mesh file for multiple GVXR runs.
        if os.path.exists(new_mesh_file):
            print(
                f"Found {new_mesh_file} so assuming conversion has already been done previously."
            )
            return new_mesh_file

        print("Converting tetrahedron mesh into triangles for GVXR")
        mesh = meshio.read(mesh_file)
        # extract np arrays of mesh data from meshio
        points = mesh.points
        tetra = mesh.get_cells_type("tetra")
        if not np.any(tetra):
            # no tetra data but trying to use tets
            err_msg = (
                "User asked to use tets but mesh file does not contain Tetrahedron data"
            )
            raise ValueError(err_msg)
        mat_ids_tet = mesh.get_cell_data("cell_tags", "tetra")
        # extract surface triangles from volume tetrahedron mesh
        elements, mat_ids = tets2tri(tetra, points, mat_ids_tet)
        cells = [("triangle", elements)]

        # convert extracted triangles into new meshio object and write out to file
        tri_mesh = meshio.Mesh(
            points,
            cells,
            # Each item in cell data must match the cells array
            cell_data={"cell_tags": [mat_ids]},
        )
        tri_mesh.cell_tags = find_the_key(mesh.cell_tags, np.unique(mat_ids))
        print(f"Saving new triangle mesh as {new_mesh_file}")
        tri_mesh.write(new_mesh_file)

        return new_mesh_file

    def warn_Nikon(use_nikon, parameter_string):
        if use_nikon:
            msg = (
                "Data is being read in from Nikon File. "
                +f"However, you have defined GVXR.{parameter_string}. " \
                +"Thus the equivalent parameter in the Nikon file will be ignored."
            )
            warning_message(msg)
        return

    def check_required_params(Parameters):
        """
        function to check the input required parameters
        and throw an error if they are not defined.

        Note: The required parameters change if a nikon
        file is used as some params are assumed to be
        set in the nikon file.

        The logic here can be tough to follow but essentially:

        If we are NOT using a nikon file we need to check if any
        listed parameters ARE NOT defined and if so throw an error.

        """
        # required regardless of nikon file
        required_params = [
            "Material_list",
        ]
        if not hasattr(Parameters, "Nikon_file"):
            # list of all prams that are required when not using a nikon file
            required_params = required_params + [
                "Beam_PosX",
                "Beam_PosY",
                "Beam_PosZ",
                "Detect_PosX",
                "Detect_PosY",
                "Detect_PosZ",
                "Pix_X",
                "Pix_Y",
                "Model_PosX",
                "Model_PosY",
                "Model_PosZ",
            ]
        for param in required_params:
            if not hasattr(Parameters, param):
                raise ValueError(f"You must Specify GVXR.{param} in Input Parameters.")
        return

    class MyConfig:
        validate_assignment = True

    # A class for holding x-ray beam data
    @dataclass(config=MyConfig)
    class Xray_Beam:
        Beam_PosX: float
        Beam_PosY: float
        Beam_PosZ: float
        Beam_Type: str
        Energy: List[float] = Field(default=None)
        Intensity: List[float] = Field(default=None)
        Tube_Voltage: float = Field(default=0.0)
        Tube_Angle: float = Field(default=12.0)
        Filter_ThicknessMM: float = Field(default=None)
        Filter_Material: str = Field(default=None)
        Beam_Pos_units: str = Field(default="mm")
        Energy_units: str = Field(default="MeV")

    @classmethod
    def xor(x: bool, y: bool) -> bool:
        """Simple function to perform xor with two bool values"""
        return bool((x and not y) or (not x and y))

    Data = {}

    #########################################
    # For reference in all cases our co-ordinates
    # are defined as:
    #
    # x -> horizontal on the detector/output image
    # y -> Vertical on the detector/output image
    # z -> axis between the src. and the detector.
    # Also projections are rotations counter
    # clockwise around the y axis.
    ##############################################
    # A class for holding x-ray detector data
    @dataclass
    class Xray_Detector:
        # position of detector in space
        Det_PosX: float
        Det_PosY: float
        Det_PosZ: float
        # number of pixels in the x and y dir
        Pix_X: int
        Pix_Y: int
        # pixel spacing
        Spacing_X: float = Field(default=0.5)
        Spacing_Y: float = Field(default=0.5)
        # units
        Det_Pos_units: str = Field(default="mm")
        Spacing_units: str = Field(default="mm")

    @dataclass
    class Cad_Model:
        # position of cad model in space
        Model_PosX: float
        Model_PosY: float
        Model_PosZ: float
        # scaling of cad model along 3 axes
        Model_ScaleX: float = Field(default=1.0)
        Model_ScaleY: float = Field(default=1.0)
        Model_ScaleZ: float = Field(default=1.0)
        # inital rotation of model around each axis [x,y,z]
        # To keep things simple this defaults to [0,0,0]
        # Nikon files define these as Tilt, InitalAngle, and Roll respectively.
        rotation: float = Field(default_factory=lambda: [0, 0, 0])
        Model_Pos_units: str = Field(default="mm")
        Model_Mesh_units: str = Field(default="mm")

    OUT_DIR = "{}/GVXR-Images".format(PROJECT_DIR)
    IN_MESH_DIR = "{}/Meshes".format(VLC.InputDir)
    OUT_MESH_DIR = "{}/Meshes".format(PROJECT_DIR)

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    for GVXRName, GVXRParams in GVXRDicts.items():
        # Perform some checks on the info in GVXRParams
        Parameters = Namespace(**GVXRParams)
        # check for required parameters
        check_required_params(Parameters)
        # check mesh for file extension and if not present assume
        # salome med
        root, ext = os.path.splitext(Parameters.mesh)
        if not ext:
            ext = ".med"
        mesh = root + ext
        # If mesh is an absolute path use it
        if os.path.isabs(mesh):
            IN_FILE = mesh
        # if not check the input Mesh directory
        elif os.path.exists(f"{IN_MESH_DIR}/{mesh}"):
            IN_FILE = "{}/{}".format(IN_MESH_DIR, mesh)
        # If not assume the file is in the output Mesh directory
        else:
            IN_FILE = "{}/{}".format(OUT_MESH_DIR, mesh)

        GVXRDict = {
            "mesh_file": IN_FILE,
            "output_file": "{}/{}".format(OUT_DIR, GVXRName),
        }
        # Define flag to display visualizations
        if mode == "Headless":
            GVXRDict["Headless"] = True
        else:
            GVXRDict["Headless"] = False

        # check to material list is valid
        GVXRDict['Material_Types'] = Check_Materials(
            Parameters.Material_list,
            getattr(Parameters,"Amounts",[]),
            getattr(Parameters,"Density",[])
            )

        GVXRDict["Material_list"] = Parameters.Material_list
        GVXRDict["Amounts"] = getattr(Parameters,"Amounts",[])
        GVXRDict["Density"] = getattr(Parameters,"Density",[])
        ########### Setup x-ray beam ##########
        # create dummy beam, detector and cad model
        # to get filled in with values either from Parameters OR Nikon file.
        dummy_Det = Xray_Detector(Det_PosX=0, Det_PosY=0, Det_PosZ=0, Pix_X=0, Pix_Y=0)
        dummy_Model = Cad_Model(Model_PosX=0, Model_PosY=0, Model_PosZ=0)
        dummy_Beam = Xray_Beam(Beam_PosX=0, Beam_PosY=0, Beam_PosZ=0, Beam_Type="point")
        #################################################
        if hasattr(Parameters, "Nikon_file"):
            if os.path.isabs(Parameters.Nikon_file):
                # if abs path use that
                # Note: we need to now convert it to a path is accessible within the container
                    if is_bound(Nikon_file,vlab_dir=VLC.VL_HOST_DIR):
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
                Nikon_file = f"{PARAMETERS_DIR}/{Parameters.Nikon_file}"

            if os.path.exists(Nikon_file):
                print(f"Reading GVXR parameters from Nikon file: {Nikon_file}")
            else:
                # convert file path from container to host if necessary so errors make sense
                Nikon_file = container_to_host_path(Nikon_file)
                raise FileNotFoundError(
                    f"Could not find Nikon file {Nikon_file}\n \
                Please check the file is in the input directory {VLC.VL_HOST_DIR}/Input/GVXR \n \
                or that path to this file is correct."
                )

            GVXRDict = ReadNikonData(
                GVXRDict, Nikon_file, dummy_Beam, dummy_Det, dummy_Model
            )
            Use_Nikon_File = True

        else:
            Use_Nikon_File = False
            GVXRDict["Beam"] = dummy_Beam
            GVXRDict["Model"] = dummy_Model
            GVXRDict["Detector"] = dummy_Det
        #############################################################
        # fill in values for x-ray detector, beam and cad model
        # from Parameters if set.

        if hasattr(Parameters, "energy_units"):
            warn_Nikon(Use_Nikon_File, "energy_units")
            GVXRDict["Beam"].Energy_units = Parameters.energy_units

        if hasattr(Parameters, "Tube_Angle"):
            warn_Nikon(Use_Nikon_File, "Tube_Angle")
            GVXRDict["Beam"].Tube_Angle = Parameters.Tube_Angle

        if hasattr(Parameters, "Tube_Voltage"):
            warn_Nikon(Use_Nikon_File, "Tube_Voltage")
            GVXRDict["Beam"].Tube_Voltage = Parameters.Tube_Voltage

        if hasattr(Parameters, "use_spekpy"):
            warn_Nikon(Use_Nikon_File, "use_spekpy")
            use_spekpy = Parameters.use_spekpy
        else:
            use_spekpy = None

        if use_spekpy == True:
            GVXRDict["Beam"] = InitSpectrum(
                Beam=dummy_Beam, Headless=GVXRDict["Headless"]
            )
        elif use_spekpy == False:
            if hasattr(Parameters, "Energy") and hasattr(Parameters, "Intensity"):
                warn_Nikon(Use_Nikon_File, "Energy")
                warn_Nikon(Use_Nikon_File, "Intensity")
                GVXRDict["Beam"].Energy = Parameters.Energy
                GVXRDict["Beam"].Intensity = Parameters.Intensity
            else:
                print(
                    "you must Specify a beam Energy and Beam Intensity when not using Spekpy."
                )
                sys.exit(1)

        # Xray Beam Position
        if hasattr(Parameters, "Beam_PosX"):
            warn_Nikon(Use_Nikon_File, "Beam_PosX")
            GVXRDict["Beam"].Beam_PosX = Parameters.Beam_PosX
        if hasattr(Parameters, "Beam_PosY"):
            warn_Nikon(Use_Nikon_File, "Beam_PosY")
            GVXRDict["Beam"].Beam_PosY = Parameters.Beam_PosY
        if hasattr(Parameters, "Beam_PosZ"):
            warn_Nikon(Use_Nikon_File, "Beam_PosZ")
            GVXRDict["Beam"].Beam_PosZ = Parameters.Beam_PosZ
        if hasattr(Parameters, "Beam_Pos_units"):
            warn_Nikon(Use_Nikon_File, "Beam_Pos_units")
            GVXRDict["Beam"].Beam_Pos_units = Parameters.Beam_Pos_units

        # Beam Type
        if hasattr(Parameters, "Beam_Type"):
            GVXRDict["Beam"].Beam_Type = Parameters.Beam_Type

        # Detector position
        if hasattr(Parameters, "Detect_PosX"):
            warn_Nikon(Use_Nikon_File, "Detect_PosX")
            GVXRDict["Detector"].Det_PosX = Parameters.Detect_PosX
        if hasattr(Parameters, "Detect_PosY"):
            warn_Nikon(Use_Nikon_File, "Detect_PosY")
            GVXRDict["Detector"].Det_PosY = Parameters.Detect_PosY
        if hasattr(Parameters, "Detect_PosZ"):
            warn_Nikon(Use_Nikon_File, "Detect_PosZ")
            GVXRDict["Detector"].Det_PosZ = Parameters.Detect_PosZ
        if hasattr(Parameters, "Detect_Pos_units"):
            warn_Nikon(Use_Nikon_File, "Detect_Pos_units")
            GVXRDict["Detector"].Det_Pos_units = Parameters.Detect_Pos_units

        # Detector Resolution
        if hasattr(Parameters, "Pix_X"):
            warn_Nikon(Use_Nikon_File, "Pix_X")
            GVXRDict["Detector"].Pix_X = Parameters.Pix_X
        if hasattr(Parameters, "Pix_Y"):
            warn_Nikon(Use_Nikon_File, "Pix_Y")
            GVXRDict["Detector"].Pix_Y = Parameters.Pix_Y

        # Detector Spacing
        if hasattr(Parameters, "SpacingX"):
            warn_Nikon(Use_Nikon_File, "SpacingX")
            GVXRDict["Detector"].Spacing_X = Parameters.SpacingX
        if hasattr(Parameters, "SpacingY"):
            warn_Nikon(Use_Nikon_File, "SpacingY")
            GVXRDict["Detector"].Spacing_Y = Parameters.SpacingY
        if hasattr(Parameters, "Spacing_units"):
            warn_Nikon(Use_Nikon_File, "Spacing_units")
            GVXRDict["Detector"].Spacing_units = Parameters.Spacing_units

        # CAD Model position
        if hasattr(Parameters, "Model_PosX"):
            GVXRDict["Model"].Model_PosX = Parameters.Model_PosX
        else:
            GVXRDict["Model"].Model_PosX = 0

        if hasattr(Parameters, "Model_PosY"):
            GVXRDict["Model"].Model_PosY = Parameters.Model_PosY
        else:
            GVXRDict["Model"].Model_PosY = 0

        if hasattr(Parameters, "Model_PosZ"):
            GVXRDict["Model"].Model_PosZ = Parameters.Model_PosZ
        else:
            GVXRDict["Model"].Model_PosZ = 0
            
        if hasattr(Parameters, "Model_Pos_units"):
            warn_Nikon(Use_Nikon_File, "Model_Pos_units")
            GVXRDict["Model"].Model_Pos_units = Parameters.Model_Pos_units
        if hasattr(Parameters, "Model_Mesh_units"):
            GVXRDict["Model"].Model_Mesh_units = Parameters.Model_Mesh_units

        # CAD Model scaling factor
        if hasattr(Parameters, "Model_ScaleX"):
            GVXRDict["Model"].Model_ScaleX = Parameters.Model_ScaleX
        if hasattr(Parameters, "Model_ScaleY"):
            GVXRDict["Model"].Model_ScaleY = Parameters.Model_ScaleY
        if hasattr(Parameters, "Model_ScaleZ"):
            GVXRDict["Model"].Model_ScaleZ = Parameters.Model_Pos_units

        if hasattr(Parameters, "num_projections"):
            warn_Nikon(Use_Nikon_File, "num_projections")
            GVXRDict["num_projections"] = Parameters.num_projections

        if hasattr(Parameters, "angular_step"):
            warn_Nikon(Use_Nikon_File, "angular_step")
            GVXRDict["angular_step"] = Parameters.angular_step

        ################################################################
        if hasattr(Parameters, "rotation"):
            GVXRDict["Model"].rotation = Parameters.rotation

        if hasattr(Parameters, "image_format"):
            GVXRDict["im_format"] = Parameters.image_format
        if hasattr(Parameters, "FFNorm"):
            GVXRDict["FFNorm"] = Parameters.FFNorm
        if hasattr(Parameters, "bitrate"):
            GVXRDict["bitrate"] = Parameters.bitrate            
        if hasattr(Parameters, "use_tetra"):
            # Convert tetrahedron data into triangles
            GVXRDict["use_tetra"] = Parameters.use_tetra

        if hasattr(Parameters, "downscale"):
            if 0.0 < float(Parameters.downscale) <= 1.0:
                GVXRDict["downscale"] = float(Parameters.downscale)
            else:
                raise ValueError(
                    f"Invalid parameter for GVXR.downscale {Parameters.downscale}, \
                            this must be between 0.0 and 1.0."
                )

        if hasattr(Parameters, "fill_value"):
            try:
                GVXRDict["fill_value"] = float(Parameters.fill_value)
            except:
                raise ValueError(
                    f"Invalid parameter for GVXR.fill_value {Parameters.fill_value}, \
                            this must be a float or convertible to a float"
                )
                
        if hasattr(Parameters, "fill_percent"):
            if 0.0 < float(Parameters.fill_percent) <= 1.0:
                GVXRDict["fill_percent"] = float(Parameters.fill_percent)
            else:
                raise ValueError(
                    f"Invalid parameter for GVXR.fill_percent {Parameters.fill_percent}, \
                            this must be greater than 0.0 and less than or equal to 1.0."
                )

        # catch any extra options and throw an error to say they are invalid
        param_dict = vars(Parameters)
        for key in ["Name", "mesh"]:
            del param_dict[key]

        diff = set(param_dict.keys()).difference(GVXR_parameters)
        if list(diff) != []:
            invalid_options = ""
            for i in list(diff):
                invalid_options = invalid_options + f"GVXR.{i}={param_dict[i]}\n"
            print(
                f"The following input parameters were not recognized for GVXR:\n{invalid_options}"
            )
            sys.exit(1)

        Param_dir = "{}/run_params/".format(PROJECT_DIR)
        if not os.path.exists(Param_dir):
            os.makedirs(Param_dir)
        pth = "{}/setup_params_{}.json".format(Param_dir, GVXRName)
        dump_to_json(GVXRDict, pth)
