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
    energy_units: str = field(default='MeV')
    Intensity: int = field(default=1000)
        
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
    Spacing_in_mm: float =field(default=0.5)

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


        GVXRDict = { 'input_file':IN_FILE,
                    'output_file':"{}/{}".format(VL.OUT_DIR,GVXRName)
                }
# Logic to handle placing greyscale file in the correct place. i.e. in the output dir not the run directory.
        if hasattr(Parameters,'Material_file') and os.path.isabs(Parameters.Material_file):
        # Abs. paths go where they say
            GVXRDict['Material_file'] = Parameters.Material_file
        elif hasattr(Parameters,'Material_file') and not os.path.isabs(Parameters.Material_file):
        # This makes a non abs. path relative to the output directory not the run directory (for consistency)
            GVXRDict['Material_file'] = "{}/{}".format(VL.OUT_DIR,Parameters.Material_file)
        else:
        # greyscale not given so generate a file in the output directory 
            GVXRDict['Material_file'] = "{}/Materials_{}.csv".format(VL.OUT_DIR,GVXRName) 
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
        if hasattr(Parameters,'Spacing_in_mm'): 
            Detector.Spacing_in_mm = Parameters.Spacing_in_mm
  
        GVXRDict['Detector'] = Detector
################################
        if hasattr(Parameters,'num_angles'): 
            GVXRDict['num_angles'] = Parameters.num_angles

        if hasattr(Parameters,'max_angles'): 
            GVXRDict['max_angles'] = Parameters.max_angles
        

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
