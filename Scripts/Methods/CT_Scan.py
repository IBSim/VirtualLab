import os
import sys

sys.dont_write_bytecode = True
from types import SimpleNamespace as Namespace

from Scripts.Common.VLParallel import VLPool
from Scripts.Common.utils import Method_base
from Scripts.Common.VLContainer import Container_Utils as Utils


class Method(Method_base):
    def Setup(self, VL, GVXRDicts, RunGVXR=True):
        """
        GVXR - Simulation of X-ray CT scans
        """
        if not (self.RunFlag and GVXRDicts):
            return
        # if called from VLsetup add dummy data and return, this is here
        # To skip setup because setup requires packages the manger does
        # not have and all VLsetup needs is a non blank dict then setup
        # proper can be called by VLModule
        if self.clsname == "VLSetup":
            self.Data = {"Master": True}
            return
        # filter out dict so we only setup runs assigned this container.
        GVXRDicts = VL.filter_runs(GVXRDicts)

        from importlib import import_module
        import copy
        import Scripts.Common.VLFunctions as VLF
        import pydantic
        from pydantic.dataclasses import dataclass, Field
        from typing import Optional, List
        from Scripts.VLPackages.GVXR.Utils_IO import ReadNikonData
        from Scripts.VLPackages.GVXR.GVXR_utils import (
            Check_Materials,
            InitSpectrum,
            dump_to_json,
        )

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
            Filters: str = Field(default=None)
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
                raise ValueError(
                    "If using Energy and/or Intenisity You must define both."
                )
            # if using speckpy you need both Tube angle and voltage
            elif xor(Voltage_defined, Angle_defined):
                raise ValueError(
                    "If using Tube Angle and/or Tube Voltage you must define both."
                )

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

        OUT_DIR = "{}/GVXR-Images".format(VL.PROJECT_DIR)
        MESH_DIR = "{}/Meshes".format(VL.PROJECT_DIR)

        if not os.path.exists(OUT_DIR):
            os.makedirs(OUT_DIR)
        self.Data = {}

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
            # If not assume the file is in the Mesh directory
            else:
                IN_FILE = "{}/{}".format(MESH_DIR, mesh)

            GVXRDict = {
                "mesh_file": IN_FILE,
                "output_file": "{}/{}".format(OUT_DIR, GVXRName),
            }
            # Define flag to display visualisations
            if VL.mode == "Headless":
                GVXRDict["Headless"] = True
            else:
                GVXRDict["Headless"] = False
            # Logic to handle placing Material file in the correct place.
            #  i.e. in the output dir not the run directory.
            if hasattr(Parameters, "Material_list"):
                Check_Materials(Parameters.Material_list)
                GVXRDict["Material_list"] = Parameters.Material_list
            else:
                raise ValueError(
                    "You must Specify a Material_list in Input Parameters."
                )

            ########### Setup x-ray beam ##########
            # create dummy beam and to get filled in with values either from Parameters OR Nikon file.
            dummy_Beam = Xray_Beam(
                Beam_PosX=0, Beam_PosY=0, Beam_PosZ=0, Beam_Type="point"
            )

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
                    raise ValueError(
                        "you must Specify a beam Energy and Beam Intensity when not using Spekpy."
                    )
            #################################################
            if hasattr(Parameters, "Nikon_file"):
                Nikon_file = Parameters.Nikon_file
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

                if hasattr(Parameters, "rotation"):
                    Model.rotation = Parameters.rotation
                GVXRDict["Model"] = Model

                if hasattr(Parameters, "num_projections"):
                    GVXRDict["num_projections"] = Parameters.num_projections

                if hasattr(Parameters, "angular_step"):
                    GVXRDict["angular_step"] = Parameters.angular_step

            ################################################################
            if hasattr(Parameters, "image_format"):
                GVXRDict["im_format"] = Parameters.image_format

            if hasattr(Parameters, "use_tetra"):
                GVXRDict["use_tetra"] = Parameters.use_tetra

            if hasattr(Parameters, "Vulkan"):
                GVXRDict["Vulkan"] = Parameters.Vulkan

            self.Data[GVXRName] = GVXRDict.copy()
            Param_dir = "{}/run_params/".format(VL.PROJECT_DIR)
            if not os.path.exists(Param_dir):
                os.makedirs(Param_dir)
            Json_file = "{}/{}_params.json".format(Param_dir, GVXRName)
            dump_to_json(self.Data[GVXRName], Json_file)

    @staticmethod
    def PoolRun(VL, AnalysisDict, **kwargs):
        """
        Function call CT_Scan with the information from AnalysisDict.

        Note: This must have the decorator @staticmethod as it does not take the
        argument 'self'.
        """
        from Scripts.VLPackages.GVXR.CT_Scan import CT_scan

        Errorfnc = CT_scan(**AnalysisDict)
        if Errorfnc:
            return Errorfnc

    def Run(self, VL, **kwargs):
        """
        Function to setup the openGL context and initiate runs for GVXR
        """
        # create openGL context for runs
        from gvxrPython3 import gvxr

        if not self.Data:
            return
        VL.Logger("\n### Starting GVXR ###\n", Print=True)
        VL.Logger(gvxr.getVersionOfSimpleGVXR())
        VL.Logger(gvxr.getVersionOfCoreGVXR())
        # Create an OpenGL context
        VL.Logger("Create an OpenGL context")
        if VL.mode == "Headless":
            # headless
            gvxr.createWindow(-1, 0, "EGL", 4, 5)
        else:
            gvxr.createWindow(-1, 1, "OPENGL", 4, 5)

        launcher = VL._Launcher
        if VL._NbJobs > 1:
            VL.Logger(
                "********************************************\n"
                "WARNING: GVXR does not work with pathos or mpi\n"
                " Thus setting NbJobs has no effect and GVX runs\n"
                " will be performed sequentially.\n"
                " To run experiments in parallel set NbJobs=1 \n"
                " and use nb_containers to use multiple containers.\n"
                "********************************************"
            )
        VL._Launcher = "sequential"

        Errorfnc = VLPool(VL, self.GetPoolRun(), self.Data)
        if Errorfnc:
            VL.Exit(
                "The following GVXR routine(s) finished with errors:\n{}".format(
                    Errorfnc
                ),
                Cleanup=False,
            )

        gvxr.destroyAllWindows()
        VL.Logger("\n### GVXR Complete ###", Print=True)

    def Spawn(self, VL, **kwargs):
        """
        This is the function called when running VirtualLab.CT_Scan() in the
        RunFile.

        ***************************************************************
        ***********  Note for using multiple containers   *************
        ***************************************************************
        This particular module runs in parallel using multiple containers.
        However, this can be very problematic and resource (particularly
        ram) intensive.

        Therefore, for running in parallel we recommend using pathos or mpi
        set via the VL._Launcher option (see VLParallel.py). However, if this
        is not an option. VirtualLab can, with some setup, spread defined
        jobs over multiple containers.

        To run in parallel with multiple containers you will need first
        need to set: Num_Cont=len(VL.container_list['#MethodName'].

        You will also need to call MethodDicts = VL.filter_runs(MethodDicts)
        during setup. This will Filter MethodDicts for runs that are defined
        in the running container. If you dont call this it will run all the jobs
        on each container which I suspect is not what you want.

        Finally you will likely want to set VL._Launcher to 'sequential' that
        is unless you really want to run jobs in parallel inside multiple
        containers, though I can't see what you would expect to gain from that.
        As the increased overhead and memory usage from multiple containers
        massively outweighs any minute performance gain.

        """

        self._SpawnBase(VL,"CT_Scan","GVXR", Num_Cont=len(VL.container_list[MethodName]), run_kwargs=kwargs) # method name and container name
 
