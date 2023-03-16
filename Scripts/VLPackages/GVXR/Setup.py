def GVXR_Setup(GVXRDicts,PROJECT_DIR,mode):
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
    from Scripts.VLPackages.GVXR.Utils_IO import ReadNikonData
    from Scripts.VLPackages.GVXR.GVXR_utils import (
        Check_Materials,
        InitSpectrum,
        dump_to_json,
    )
    import VLconfig as VLC

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
        Tube_Voltage: float = Field(default=None)
        Tube_Angle: float = Field(default=None)
        Filter_ThicknessMM: float = Field(default=None)
        Filter_Material: str = Field(default=None)
        Beam_Pos_units: str = Field(default="mm")
        Energy_units: str = Field(default="MeV")

    @classmethod
    def xor(x: bool, y: bool) -> bool:
        """Simple function to perform xor with two bool values"""
        return bool((x and not y) or (not x and y))

    @pydantic.root_validator(pre=True)
    @classmethod
    def check_BeamEnergy_or_TubeVoltage(cls, values):
        """
        if defining own values you need both Energy and Intensity
        xor here passes if both or neither are defined.
        The same logic also applies if using Spekpy with tube Voltage and angle.
        This prevents the user from defining only one of the two needed values.
        """

        Energy_defined = "Energy" not in values
        Intensity_defined = "Intensity" not in values
        Voltage_defined = "Tube_Voltage" not in values
        Angle_defined = "Tube_Angle" not in values

        if xor(Energy_defined, Intensity_defined):
            print("If using Energy and/or Intenisity. You must define both in the input parameter's."
            )
            sys.exit(1)
        # if using speckpy you need both Tube angle and voltage
        elif xor(Voltage_defined, Angle_defined):
            print("If using Tube Angle and/or Tube Voltage You must define both in the input parameter's.")
            sys.exit(1)
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

    OUT_DIR = "{}/GVXR-Images".format(PROJECT_DIR)
    IN_MESH_DIR ="{}/Meshes".format(VLC.InputDir)
    OUT_MESH_DIR = "{}/Meshes".format(PROJECT_DIR)

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    for GVXRName, GVXRParams in GVXRDicts.items():
        # Perform some checks on the info in GVXRParams
        Parameters = Namespace(**GVXRParams)
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
        elif os.path.exists(f'{IN_MESH_DIR}/{mesh}'):
            IN_FILE = "{}/{}".format(IN_MESH_DIR, mesh)
        # If not assume the file is in the output Mesh directory
        else:
            IN_FILE = "{}/{}".format(OUT_MESH_DIR, mesh)

        GVXRDict = {
            "mesh_file": IN_FILE,
            "output_file": "{}/{}".format(OUT_DIR, GVXRName),
        }
        # Define flag to display visualisations
        if mode == "Headless":
            GVXRDict["Headless"] = True
        else:
            GVXRDict["Headless"] = False
        # Logic to handle placing Material file in the correct place.
        #  i.e. in the output dir not the run directory.
        if hasattr(Parameters, "Material_list"):
            Check_Materials(Parameters.Material_list)
            GVXRDict["Material_list"] = Parameters.Material_list
        else:
            print("You must Specify a Material_list in Input Parameters.")
            sys.exit(1)

        ########### Setup x-ray beam ##########
        # create dummy beam and to get filled in with values either from Parameters OR Nikon file.
        dummy_Beam = Xray_Beam(
            Beam_PosX=0, Beam_PosY=0, Beam_PosZ=0, Beam_Type="point"
        )
        #################################################
        if hasattr(Parameters, "Nikon_file"):
            if os.path.isabs(Parameters.Nikon_file):
            #if abs path use that
                Nikon_file = Parameters.Nikon_file
            else:
                Nikon_file = f'{VLC.InputDir}/GVXR/{Parameters.Nikon_file}'
            # if not check the input directory
            if os.path.exists(Nikon_file):
                print(f"Reading GVXR parameters from Nikon file: {Nikon_file}")
            else:
                print(f"Could not find Nikon file {Nikon_file}\n",
                f"Please check the file is in the input directory {VLC.VL_HOST_DIR}/Input/GVXR \n",
                "or that path to this file is correct.")
                sys.exit(1)
            
            # create dummy detector and cad model to get filled in with values
            # from nikon file.
            # Note: the function adds the 3 data classes to the GVXRdict itself.
            dummy_Det = Xray_Detector(
                Det_PosX=0, Det_PosY=0, Det_PosZ=0, Pix_X=0, Pix_Y=0
            )
            dummy_Model = Cad_Model(Model_PosX=0, Model_PosY=0, Model_PosZ=0)
            GVXRDict = ReadNikonData(
                GVXRDict, Nikon_file, dummy_Beam, dummy_Det, dummy_Model
            )
        else:
                #############################################################
                # fill in values for x-ray detector, beam and cad model
                # from Parameters.
            if hasattr(Parameters, "Energy_units"):
                dummy_Beam.Energy_units = Parameters.Energy_units

            if hasattr(Parameters, "Tube_Angle"):
                dummy_Beam.Tube_Angle = Parameters.Tube_Angle
            if hasattr(Parameters, "Tube_Voltage"):
                dummy_Beam.Tube_Voltage = Parameters.Tube_Voltage

            if Parameters.use_spekpy:
                dummy_Beam = InitSpectrum(
                    Beam=dummy_Beam, Headless=GVXRDict["Headless"]
                )
            else:
                if hasattr(Parameters, "Energy") and hasattr(Parameters, "Intensity"):
                    dummy_Beam.Energy = Parameters.Energy
                    dummy_Beam.Intensity = Parameters.Intensity
                else:
                    print("you must Specify a beam Energy and Beam Intensity when not using Spekpy.")
                    sys.exit(1)
                    
            dummy_Beam.Beam_PosX = Parameters.Beam_PosX
            dummy_Beam.Beam_PosY = Parameters.Beam_PosY
            dummy_Beam.Beam_PosZ = Parameters.Beam_PosZ
            dummy_Beam.Beam_Type = Parameters.Beam_Type
            if hasattr(Parameters, "Beam_Pos_units"):
                dummy_Beam.Pos_units = Parameters.Beam_Pos_units
            GVXRDict["Beam"] = dummy_Beam

            Detector = Xray_Detector(
                Det_PosX=Parameters.Detect_PosX,
                Det_PosY=Parameters.Detect_PosY,
                Det_PosZ=Parameters.Detect_PosZ,
                Pix_X=Parameters.Pix_X,
                Pix_Y=Parameters.Pix_Y,
            )
            if hasattr(Parameters, "Detect_Pos_units"):
                Detector.Pos_units = Parameters.Detect_Pos_units
            if hasattr(Parameters, "Spacing_X"):
                Detector.Spacing_X = Parameters.Spacing_X

            if hasattr(Parameters, "Spacing_Y"):
                Detector.Spacing_Y = Parameters.Spacing_Y

            GVXRDict["Detector"] = Detector

            Model = Cad_Model(
                Model_PosX=Parameters.Model_PosX,
                Model_PosY=Parameters.Model_PosY,
                Model_PosZ=Parameters.Model_PosZ,
            )

            # add in model scaling factor
            if hasattr(Parameters, "Model_ScaleX"):
                Model.Model_ScaleX = Parameters.Model_ScaleX

            if hasattr(Parameters, "Model_ScaleY"):
                Model.Model_ScaleY = Parameters.Model_ScaleY

            if hasattr(Parameters, "Model_ScaleZ"):
                Model.Model_ScaleZ = Parameters.Model_Pos_units

            if hasattr(Parameters, "Model_Pos_units"):
                Model.Model_Pos_units = Parameters.Model_Pos_units

            # if hasattr(Parameters, "rotation"):
            #     Model.rotation = Parameters.rotation
            GVXRDict["Model"] = Model

            if hasattr(Parameters, "num_projections"):
                GVXRDict["num_projections"] = Parameters.num_projections

            if hasattr(Parameters, "angular_step"):
                GVXRDict["angular_step"] = Parameters.angular_step

        ################################################################
        if hasattr(Parameters, "rotation"):
            Model.rotation = Parameters.rotation

        if hasattr(Parameters, "image_format"):
            GVXRDict["im_format"] = Parameters.image_format

        if hasattr(Parameters, "use_tetra"):
            GVXRDict["use_tetra"] = Parameters.use_tetra

        if hasattr(Parameters, "downscale"):
            if 0.0 < float(Parameters.downscale) <= 1.0:
                GVXRDict["downscale"] = float(Parameters.downscale)
            else:
                raise ValueError(f"Invalid parameter for GVXR.downscale {Parameters.downscale}, \
                            this must be between 0.0 and 1.0.")

        Param_dir = "{}/run_params/".format(PROJECT_DIR)
        if not os.path.exists(Param_dir):
            os.makedirs(Param_dir)
        pth = "{}/setup_params_{}.json".format(Param_dir,GVXRName)
        dump_to_json(GVXRDict, pth)