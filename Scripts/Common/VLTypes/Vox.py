import os
import sys
sys.dont_write_bytecode=True
from types import SimpleNamespace as Namespace
from importlib import import_module
import copy
import cad2vox

import Scripts.Common.VLFunctions as VLF


def Check_Threads(num_threads):
    """ Function to check the user defined Number of OpenMP threads are vaild"""
    try:
        int(num_threads)
    except ValueError:
        print(num_threads)
        raise ValueError("Invalid number of threads for Cad2Vox, must be an Integer value, "
        "or castable to and Integer value")

    if ((int(num_threads) < 0)):
        raise ValueError("Invalid Number of threads for Cad2Vox. Must be greater than 0")

def Setup(VL, RunVox=True):
    '''
    Vox - Mesh Voxelisation using Cuda or OpenMP
    '''
    VL.OUT_DIR = "{}/Voxel-Images".format(VL.PROJECT_DIR)
    VL.MESH_DIR = "{}/Meshes".format(VL.PROJECT_DIR)

    if not os.path.exists(VL.OUT_DIR):
        os.makedirs(VL.OUT_DIR)

    VL.VoxData = {}
    VoxDicts = VL.CreateParameters(VL.Parameters_Master, VL.Parameters_Var,'Vox')

    # if RunVox is False or VoxDicts is empty dont perform voxelisation and return instead.
    if not (RunVox and VoxDicts): return
    #for VoxName, VoxParams in VoxDicts.items():
    for I,VoxName in enumerate(VoxDicts.keys()):
        if I>0:
        #This logic allows us to append a sting to the end of output files if using more than one mesh
        # This way we dont write to the same output file if using multiple inputs.
            J="_"+I
        else:
            J=""
        VoxParams = VoxDicts[VoxName]
        Parameters = Namespace(**VoxParams)
        #check name for file extension and if not present assume salome med
        root, ext = os.path.splitext(VoxName)
        if not ext:
            ext = '.med'
        VoxName = root + ext
        # If VoxName is an absolute path use it  
        if os.path.isabs(VoxName):
            IN_FILE = VoxName
            OUT_FILE = "{}{}".format(root,J)
        # If not assume the file is in the Mesh directory
        else:
            IN_FILE="{}/{}".format(VL.MESH_DIR, VoxName)
            OUT_FILE="{}/{}{}".format(VL.OUT_DIR, root,J)
        VoxDict = { 'input_file':IN_FILE,
                    'output_file':OUT_FILE
                }
        # handle optional arguments
        if hasattr(Parameters,'unit_length'): 
            VoxDict['unit_length'] = Parameters.unit_length

        if hasattr(Parameters,'gridsize'): 
            VoxDict['gridsize'] = Parameters.Gridsize
# Logic to handle placing greyscale file in the correct place that is ion the output dir not the run directory.
        if hasattr(Parameters,'greyscale_file') and os.path.isabs(Parameters.greyscale_file):
        # Abs. paths go where they say
            VoxDict['greyscale_file'] = Parameters.greyscale_file
        elif hasattr(Parameters,'greyscale_file') and not os.path.isabs(Parameters.greyscale_file):
        # This makes a non abs. path relative to the output directory not the run directory (for consistency)
            VoxDict['greyscale_file'] = "{}/{}".format(VL.OUT_DIR,Parameters.greyscale_file)
        else:
        # greyscale not given so generate a file in the output directory 
            VoxDict['greyscale_file'] = "{}/greyscale.csv".format(VL.OUT_DIR) 

        if hasattr(Parameters,'use_tetra'): 
            VoxDict['use_tetra'] = Parameters.use_tetra

        if hasattr(Parameters,'cpu'): 
            VoxDict['cpu'] = Parameters.cpu

        if hasattr(Parameters,'solid'): 
            VoxDict['solid'] = Parameters.solid
        
        if hasattr(Parameters, 'Num_Threads'):
            Check_Threads(Parameters.Num_Threads)
            os.environ["OMP_NUM_THREADS"]=Parameters.Num_Threads

        if hasattr(Parameters,'image_format'): 
            VoxDict['im_format'] = Parameters.image_format
            
        VL.VoxData[VoxName] = VoxDict.copy()

def Run(VL):
    if not VL.VoxData: return
    print(VL.VoxData)
    VL.Logger('\n### Starting Voxelisation ###\n', Print=True)

    for item in VL.VoxData:
        Errorfnc = cad2vox.voxelise(**item)
        if Errorfnc:
            VL.Exit(VLF.ErrorMessage("The following Cad2Vox routine(s) finished with errors:\n{}".format(Errorfnc)))

    VL.Logger('\n### Voxelisation Complete ###',Print=True)
