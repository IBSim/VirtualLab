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

    for VoxName, VoxParams in VoxDicts.items():
        Parameters = Namespace(**VoxParams)
        VoxDict = { 'input_file':"{}/{}.med".format(VL.MESH_DIR, VoxName),
                    'output_file':"{}/{}.Tiff".format(VL.OUT_DIR, VoxName)
                }

        # handle optional arguments
        if hasattr(Parameters,'unit_length'): 
            VoxDict['unit_length'] = Parameters.unit_length

        if hasattr(Parameters,'gridsize'): 
            VoxDict['gridsize'] = Parameters.Gridsize

        if hasattr(Parameters,'greyscale_file'): 
            VoxDict['greyscale_file'] = Parameters.greyscale_file

        if hasattr(Parameters,'use_tetra'): 
            VoxDict['use_tetra'] = Parameters.use_tetra

        if hasattr(Parameters,'cpu'): 
            VoxDict['cpu'] = Parameters.cpu

        if hasattr(Parameters,'solid'): 
            VoxDict['solid'] = Parameters.solid
        
        if hasattr(Parameters, 'Num_Threads'):
            Check_Threads(Parameters.Num_Threads)
            os.environ["OMP_NUM_THREADS"]=str('Num_Threads')

        VL.VoxData[VoxName] = VoxDict.copy()

def Run(VL):
    if not VL.VoxData: return

    VL.Logger('\n### Starting Voxelisation ###\n', Print=True)

    for item in VL.VoxData:
        Errorfnc = cad2vox.voxelise(**item)
        if Errorfnc:
            VL.Exit(VLF.ErrorMessage("The following Cad2Vox routine(s) finished with errors:\n{}".format(Errorfnc)))

    VL.Logger('\n### Voxelisation Complete ###',Print=True)
