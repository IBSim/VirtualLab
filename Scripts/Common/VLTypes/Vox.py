import os
import sys
sys.dont_write_bytecode=True
from types import SimpleNamespace as Namespace
from importlib import import_module
import copy
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
    VoxDicts = VL.CreateParameters(VL.Parameters_Master, VL.Parameters_Var,'Vox')
    # if RunVox is False or VoxDicts is empty dont perform voxelisation and return instead.
    if not (RunVox and VoxDicts): return
    VL.VoxData = {}
    OUT_DIR = "{}/Voxel-Images".format(VL.PROJECT_DIR)
    MESH_DIR = "{}/Meshes".format(VL.PROJECT_DIR)

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    for VoxName, VoxParams in VoxDicts.items():
        Parameters = Namespace(**VoxParams)
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
            IN_FILE="{}/{}".format(MESH_DIR, mesh)


        VoxDict = { 'input_file':IN_FILE,
                    'output_file':"{}/{}".format(OUT_DIR,VoxName)
                }
        # handle optional arguments
        if hasattr(Parameters,'unit_length'): 
            VoxDict['unit_length'] = Parameters.unit_length

        if hasattr(Parameters,'gridsize'): 
            VoxDict['gridsize'] = Parameters.gridsize
# Logic to handle placing greyscale file in the correct place. i.e. in the output dir not the run directory.
        if hasattr(Parameters,'greyscale_file') and os.path.isabs(Parameters.greyscale_file):
        # Abs. paths go where they say
            VoxDict['greyscale_file'] = Parameters.greyscale_file
        elif hasattr(Parameters,'greyscale_file') and not os.path.isabs(Parameters.greyscale_file):
        # This makes a non abs. path relative to the output directory not the run directory (for consistency)
            VoxDict['greyscale_file'] = "{}/{}".format(OUT_DIR,Parameters.greyscale_file)
        else:
        # greyscale not given so generate a file in the output directory 
            VoxDict['greyscale_file'] = "{}/greyscale_{}.csv".format(OUT_DIR,VoxName) 

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
    import cad2vox
    if not VL.VoxData: return
    VL.Logger('\n### Starting Voxelisation ###\n', Print=True)

    for key in VL.VoxData.keys():
        Errorfnc = cad2vox.voxelise(**VL.VoxData[key])
        if Errorfnc:
            VL.Exit(VLF.ErrorMessage("The following Cad2Vox routine(s) finished with errors:\n{}".format(Errorfnc)))

    VL.Logger('\n### Voxelisation Complete ###',Print=True)
