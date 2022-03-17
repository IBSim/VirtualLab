import os
import h5py
import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
import scipy.stats as stats

from VLFunctions import ReadData, ReadParameters
from Scripts.Common.ML import ML
from Scripts.Common.ML.slsqp_multi import slsqp_multi

dtype = 'float64' # float64 is more accurate for optimisation purposes
torch_dtype = getattr(torch,dtype)
MaxCholeskySize = 1500

DispX = DispY = [-0.01,0.01]
DispZ = [0.01,0.02]
CoolantP, CoolantT, CoolantV = [0.4,1.6], [30,70], [5,15]
Current = [600,1000]

InTag = ['Coil X','Coil Y','Coil Z', 'Coolant Pressure',
         'Coolant Temperature', 'Coolant Velocity','Coil Current']
OutTag = ['TC_{}'.format(j) for j in range(7)]

bounds = np.transpose([DispX,DispY,DispZ,CoolantP,CoolantT,CoolantV,Current])

InputScaler = np.array([bounds[0],bounds[1] - bounds[0]])

SurfaceNormals = np.array([['TileFront', 'NX'], ['TileBack', 'NX'], ['TileSideA', 'NY'],
                          ['TileSideB', 'NY'], ['TileTop', 'NZ'],
                          ['BlockFront', 'NX'], ['BlockBack', 'NX'], ['BlockSideA', 'NY'],
                          ['BlockSideB', 'NY'],['BlockBottom', 'NZ'], ['BlockTop', 'NZ']])

# ==============================================================================
# Functions for gathering necessary data and writing to file
def MLMapping(Dir,surface):
    datapkl = "{}/SurfaceTemps.pkl".format(Dir)
    DataDict = ReadData(datapkl)
    paramfile = "{}/Parameters.py".format(Dir)
    Parameters = ReadParameters(paramfile)

    Coolant = [Parameters.Coolant[n] for n in ['Pressure','Temperature','Velocity']]
    In = [*Parameters.CoilDisplacement,*Coolant,Parameters.Current]
    Out = DataDict[surface]
    return In, Out

def CompileData(VL,DADict):
    Parameters = DADict["Parameters"]

    DataFile_path = "{}/{}".format(VL.PROJECT_DIR,Parameters.DataFile)
    InputName = getattr(Parameters,'InputName','Input')

    CmpData = Parameters.CompileData
    if type(CmpData)==str:CmpData = [CmpData]
    ResDirs = ["{}/{}".format(VL.PROJECT_DIR,resname) for resname in CmpData]

    for surface in SurfaceNormals[:,0]:
        try:
            InData, OutData = ML.CompileData(ResDirs,MLMapping,args=[surface])
            OutputName = surface
            ML.WriteMLdata(DataFile_path, CmpData, InputName,
                           OutputName, InData, OutData)
        except KeyError:
            pass
