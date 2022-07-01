import os
import sys
sys.dont_write_bytecode=True
from types import SimpleNamespace as Namespace
from importlib import import_module
import copy
import Scripts.Common.VLFunctions as VLF
from dataclasses import dataclass, field

# A class for holding x-ray beam data
@dataclass
class Xray_Beam:
    PosX: float
    PosY: float
    PosZ: float
    beam_type: str
    energy: float
    Pos_units: str = field(default='mm')
    energy_units: str = field(default='MeV')
    Intensity: int = field(default=1000)
#########################################
# For reference in all cases or co-ordiantes 
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
    PosX: float
    PosY: float
    PosZ: float
# number of pixels in the x and y dir
    Pix_X: int
    Pix_Y: int
#pixel spacing
    Spacing_X: float =field(default=0.5)
    Spacing_Y: float =field(default=0.5)
#units
    Pos_units: str = field(default='mm')
    Spacing_units: str = field(default='mm')

@dataclass
class Cad_Model:
    # position of cad model in space 
    PosX: float
    PosY: float
    PosZ: float
    # inital rotation of model around each axis [x,y,z]
    # To keep things simple this defaults to [0,0,0]
    # Nikon files define these as Tilt, InitalAngle, and Roll repectivley. 
    rotation: float  = field(default_factory=lambda:[0,0,0])
def Setup(VL, RunGVXR=True):
    '''
    GVXR - Simulation of X-ray CT scans 
    '''
    #VL.OUT_DIR = "{}/GVXR-Images".format(VL.PROJECT_DIR)
    VL.MESH_DIR = "{}/Meshes".format(VL.PROJECT_DIR)

    if not os.path.exists(VL.OUT_DIR):
        os.makedirs(VL.OUT_DIR)
    VL.GVXRData = {}
    GVXRDicts = VL.CreateParameters(VL.Parameters_Master, VL.Parameters_Var,'GVXR')
    # if RunGVXR is False or GVXRDicts is empty dont perform Simulation and return instead.
    if not (RunGVXR and GVXRDicts): return
    for GVXRName, GVXRParams in GVXRDicts.items():
        Parameters = Namespace(**GVXRParams)
        #check mesh for file extension and if not present assume salome med
        root, ext = os.path.splitext(Parameters.mesh)
        if not ext:
            ext = '.med'
        mesh = root + ext
        # If mesh is an absolute path use it  
        if os.path.isabs(mesh):
            IN_FILE = mesh
        # If not assume the file is in the Mesh directory
        else:
            IN_FILE="{}/{}".format(VL.MESH_DIR, mesh)


        GVXRDict = { 'mesh_file':IN_FILE,
                    'output_file':"{}/{}".format(VL.OUT_DIR,GVXRName)
                }
# Logic to handle placing Materail file in the correct place. i.e. in the output dir not the run directory.
        if hasattr(Parameters,'Material_file') and os.path.isabs(Parameters.Material_file):
        # Abs. paths go where they say
            GVXRDict['Material_file'] = Parameters.Material_file
        elif hasattr(Parameters,'Material_file') and not os.path.isabs(Parameters.Material_file):
        # This makes a non abs. path relative to the output directory not the run directory (for consistency)
            GVXRDict['Material_file'] = "{}/{}".format(VL.OUT_DIR,Parameters.Material_file)
        else:
        # greyscale not given so generate a file in the output directory 
            GVXRDict['Material_file'] = "{}/Materials_{}.csv".format(VL.OUT_DIR,GVXRName) 

        if hasattr(Parameters,'Nikon_file')
# create dummy beams and detector to get filled in with values if using nikon file.
            dummy_Beam = Xray_Beam(PosX=0,PosY=0,PosZ=0,beam_type='point',energy=0)
            dummy_Det = Xray_Detector(PosX=0,PosY=0,PosZ=0,Pix_X=0,Pix_Y=0)
            dummy_Model = Cad_Model(PosX=0,PosY=0,PosZ=0)
            GVXRDict = ReadNikonData(GVXRDict,dummy_Beam,dummy_Det,dummy_Model)
        
        
##########
######### create a Xray_Beam object to pass in
        Beam = Xray_Beam(PosX=Parameters.Beam_PosX,PosY=Parameters.Beam_PosY,
            PosZ=Parameters.Beam_PosZ,beam_type=Parameters.beam_type,
            energy=Parameters.energy)

        if hasattr(Parameters,'energy_units'): 
            Beam.energy_units = Parameters.energy_units

        if hasattr(Parameters,'Intensity'): 
            Beam.Intensity = Parameters.Intensity   
        GVXRDict['Beam'] = Beam
################################
######### create a Xray_Detector object to pass in
        Detector = Xray_Detector(PosX=Parameters.Detect_PosX,
            PosY=Parameters.Detect_PosY,PosZ=Parameters.Detect_PosZ,
            Pix_X=Parameters.Pix_X,Pix_Y=Parameters.Pix_Y)

        #if hasattr(Parameters,'Headless'):
        if hasattr(Parameters,'Spacing_X'): 
            Detector.Spacing_X = Parameters.Spacing_X

        if hasattr(Parameters,'Spacing_Y'): 
            Detector.Spacing_Y = Parameters.Spacing_Y   
  
        GVXRDict['Detector'] = Detector
################################
################################
########### Create Cad model to pass in
        Model = Cad_Model(PosX=Parameters.Model_PosX,
            PosY=Parameters.Model_PosY,PosZ=Parameters.Model_PosZ)
        if hasattr(Parameters,'model_rotation'):
            Model.rotation = Parameters.model_rotation
        GVXRDict['Model'] = Model
#############################################

        if hasattr(Parameters,'num_projections'): 
            GVXRDict['num_projections'] = Parameters.num_projections

        if hasattr(Parameters,'angular_step'): 
            GVXRDict['angular_step'] = Parameters.angular_step
        

        if hasattr(Parameters,'image_format'): 
            GVXRDict['im_format'] = Parameters.image_format
        if (VL.mode=='Headless'):
            GVXRDict['Headless'] = True
        VL.GVXRData[GVXRName] = GVXRDict.copy()

def Run(VL):
    from Scripts.Common.VLPackages.GVXR.CT_Scan import CT_scan
    if not VL.GVXRData: return
    VL.Logger('\n### Starting GVXR ###\n', Print=True)
    
    for key in VL.GVXRData.keys():
        Errorfnc = CT_scan(**VL.GVXRData[key])
        if Errorfnc:
            VL.Exit(VLF.ErrorMessage("The following GVXR routine(s) finished with errors:\n{}".format(Errorfnc)))

    VL.Logger('\n### GVXR Complete ###',Print=True)
