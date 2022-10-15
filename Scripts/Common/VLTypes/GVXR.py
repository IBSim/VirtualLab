def Setup(VL, RunGVXR=True):
    '''
    GVXR - Simulation of X-ray CT scans 
    '''
    # if RunGVXR is False or GVXRDicts is empty dont perform Simulation and return instead.
    GVXRDicts = VL.CreateParameters(VL.Parameters_Master, VL.Parameters_Var,'GVXR')
    if not (RunGVXR and GVXRDicts): return
    import os
    import sys
    sys.dont_write_bytecode=True
    from types import SimpleNamespace as Namespace
    from importlib import import_module
    import copy
    import Scripts.Common.VLFunctions as VLF
    import pydantic
    #from pydantic import ValidationError
    from pydantic.dataclasses import dataclass, Field
    from typing import Optional, List
    from Scripts.Common.VLPackages.GVXR.Utils_IO import ReadNikonData
    from Scripts.Common.VLPackages.GVXR.GVXR_utils import Check_Materials, InitSpectrum, dump_to_json

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
        Beam_Pos_units: str = Field(default='mm')
        Energy_units: str = Field(default='MeV')


    @classmethod
    def xor(x: bool, y: bool) -> bool:
        ''' Simple function to perform xor with two bool values'''
        return bool((x and not y) or (not x and y))

    @pydantic.root_validator(pre=True)
    @classmethod
    def check_BeamEnergy_or_TubeVoltage(cls,values):
        ''' if defining own values you need both Energy and Intensity
        xor here passes if both or neither are defined. 
        The same logic also applies if using Spekpy with tube Voltage and angle. 
        This prevents the user from defining only one of the two needed values. '''

        Energy_defined = 'Energy' not in values
        Intensity_defined  = 'Intensity' not in values
        Voltage_defined = 'Tube_Voltage' not in values
        Angle_defined = 'Tube_Angle' not in values

        if xor(Energy_defined,Intensity_defined):
            raise ValueError ('If using Energy and/or Intenisity You must define both.')
        # if using speckpy you need both Tube angle and voltage
        elif xor(Voltage_defined,Angle_defined):  
            raise ValueError('If using Tube Angle and/or Tube Voltage you must define both.')
            
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
    #pixel spacing
        Spacing_X: float =Field(default=0.5)
        Spacing_Y: float =Field(default=0.5)
    #units
        Det_Pos_units: str = Field(default='mm')
        Spacing_units: str = Field(default='mm')

    @dataclass
    class Cad_Model:
        # position of cad model in space 
        Model_PosX: float
        Model_PosY: float
        Model_PosZ: float
        # inital rotation of model around each axis [x,y,z]
        # To keep things simple this defaults to [0,0,0]
        # Nikon files define these as Tilt, InitalAngle, and Roll respectively. 
        rotation: float  = Field(default_factory=lambda:[0,0,0])
        Model_Pos_units: str = Field(default='mm')


    OUT_DIR = "{}/GVXR-Images".format(VL.PROJECT_DIR)
    MESH_DIR = "{}/Meshes".format(VL.PROJECT_DIR)

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    VL.GVXRData = {}
    
    
    for GVXRName, GVXRParams in GVXRDicts.items():
        Parameters = Namespace(**GVXRParams)
        #check mesh for file extension and if not present assume 
        # salome med
        root, ext = os.path.splitext(Parameters.mesh)
        if not ext:
            ext = '.med'
        mesh = root + ext
        # If mesh is an absolute path use it  
        if os.path.isabs(mesh):
            IN_FILE = mesh
        # If not assume the file is in the Mesh directory
        else:
            IN_FILE="{}/{}".format(MESH_DIR, mesh)


        GVXRDict = { 'mesh_file':IN_FILE,
                    'output_file':"{}/{}".format(OUT_DIR,GVXRName)
                }
# Define flag to display visualisations
        if (VL.mode=='Headless'):
            GVXRDict['Headless'] = True
        else:
            GVXRDict['Headless'] = False
# Logic to handle placing Materail file in the correct place. i.e. in the output dir not the run directory.
        if hasattr(Parameters,'Material_list'):
            Check_Materials(Parameters.Material_list)
            GVXRDict['Material_list'] = Parameters.Material_list
        else:
            raise ValueError('You must Specify a Material_list in Input Parameters.')
        
########### Setup x-ray beam ##########
# create dummy beam and to get filled in with values either from Parameters OR Nikon file.
        dummy_Beam = Xray_Beam(Beam_PosX=0,Beam_PosY=0,Beam_PosZ=0,Beam_Type='point')
        
        if hasattr(Parameters,'Energy_units'): 
            dummy_Beam.Energy_units = Parameters.Energy_units

        if hasattr(Parameters,'Tube_Angle'): 
            dummy_Beam.Tube_Angle = Parameters.Tube_Angle
        if hasattr(Parameters,'Tube_Voltage'): 
            dummy_Beam.Tube_Voltage = Parameters.Tube_Voltage 

        if Parameters.use_spekpy:
            dummy_Beam = InitSpectrum(Beam=dummy_Beam,Headless=GVXRDict['Headless'])
        else:
            if hasattr(Parameters,'Energy') and hasattr(Parameters,'Intensity'):
                dummy_Beam.Energy = Parameters.Energy
                dummy_Beam.Intensity = Parameters.Intensity
            else:
                raise ValueError('you must Specify a beam Energy and Beam Intensity when not using Spekpy.')
#################################################
        if hasattr(Parameters,'Nikon_file'):
            Nikon_file = Parameters.Nikon_file
# create dummy detector and cad model to get filled in with values 
# from nikon file.
# Note: the function adds the 3 data classes to the GVXRdict itself.
            dummy_Det = Xray_Detector(Det_PosX=0,Det_PosY=0,Det_PosZ=0,Pix_X=0,Pix_Y=0)
            dummy_Model = Cad_Model(Model_PosX=0,Model_PosY=0,Model_PosZ=0)
            GVXRDict = ReadNikonData(GVXRDict,Nikon_file,dummy_Beam,dummy_Det,dummy_Model)
        else:
#############################################################
# fill in values for x-ray detector, beam and cad model 
# from Parameters.
            dummy_Beam.Beam_PosX=Parameters.Beam_PosX
            dummy_Beam.Beam_PosY=Parameters.Beam_PosY
            dummy_Beam.Beam_PosZ=Parameters.Beam_PosZ
            dummy_Beam.Beam_Type=Parameters.Beam_Type
            if hasattr(Parameters,'Beam_Pos_units'):
                dummy_Beam.Pos_units = Parameters.Beam_Pos_units
            GVXRDict['Beam'] = dummy_Beam

            Detector = Xray_Detector(Det_PosX=Parameters.Detect_PosX,
                Det_PosY=Parameters.Detect_PosY,Det_PosZ=Parameters.Detect_PosZ,
                Pix_X=Parameters.Pix_X,Pix_Y=Parameters.Pix_Y)
            if hasattr(Parameters,'Detect_Pos_units'):
                Detector.Pos_units = Parameters.Detect_Pos_units
            if hasattr(Parameters,'Spacing_X'): 
                Detector.Spacing_X = Parameters.Spacing_X

            if hasattr(Parameters,'Spacing_Y'): 
                Detector.Spacing_Y = Parameters.Spacing_Y   
  
            GVXRDict['Detector'] = Detector

            Model = Cad_Model(Model_PosX=Parameters.Model_PosX,
                Model_PosY=Parameters.Model_PosY,Model_PosZ=Parameters.Model_PosZ)
            if hasattr(Parameters,'Model_Pos_units'):
                Model.Pos_units = Parameters.Model_Pos_units    
            
            if hasattr(Parameters,'rotation'):
                Model.rotation = Parameters.rotation
            GVXRDict['Model'] = Model

            if hasattr(Parameters,'num_projections'): 
                GVXRDict['num_projections'] = Parameters.num_projections

            if hasattr(Parameters,'angular_step'): 
                GVXRDict['angular_step'] = Parameters.angular_step
        
################################################################
        if hasattr(Parameters,'image_format'): 
            GVXRDict['im_format'] = Parameters.image_format

        if hasattr(Parameters,'use_tetra'): 
            GVXRDict['use_tetra'] = Parameters.use_tetra

        if hasattr(Parameters,'Vulkan'): 
            GVXRDict['Vulkan'] = Parameters.Vulkan

        VL.GVXRData[GVXRName] = GVXRDict.copy()
        Param_dir = "{}/run_params/".format(VL.PROJECT_DIR)
        if not os.path.exists(Param_dir):
            os.makedirs(Param_dir)
        Json_file = "{}/{}_params.json".format(Param_dir,GVXRName)
        dump_to_json(VL.GVXRData[GVXRName],Json_file)

def Run(VL,run_ids=None):
    from Scripts.Common.VLPackages.GVXR.CT_Scan import CT_scan
    from gvxrPython3 import gvxr
    if not VL.GVXRData: return
    VL.Logger('\n### Starting GVXR ###\n', Print=True)
    print (gvxr.getVersionOfSimpleGVXR())
    print (gvxr.getVersionOfCoreGVXR())
    # Create an OpenGL context
    print("Create an OpenGL context")
    if VL.mode=='Headless':
    #headless
        gvxr.createWindow(-1,0,"EGL",4,5);
    else:
        gvxr.createWindow(-1,1,"OPENGL",4,5);
        #gvxr.setWindowSize(512, 512); 
    
    # if given a subset of runs extract only those runs
    print(run_ids)
    if run_ids:
        all_runs = list(VL.GVXRData.keys())
        run_list = [all_runs[i] for i in run_ids]
    else:
        run_list = list(VL.GVXRData.keys())

    for key in run_list:
        Errorfnc = CT_scan(**VL.GVXRData[key])
        if Errorfnc:
            VL.Exit("The following GVXR routine(s) finished with errors:\n{}".format(Errorfnc))
    gvxr.destroyAllWindows()
    VL.Logger('\n### GVXR Complete ###',Print=True)
