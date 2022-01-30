import os
import h5py
import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt

from Functions import DataScale, DataRescale, GetResPaths, ReadData, ReadParameters, Writehdf
from Optimise import FuncOpt
from MLModels import ExactGPmodel



dtype = 'float64' # float64 is more accurate for optimisation purposes
torch_dtype = getattr(torch,dtype)


def CompileData(DADict,ResPaths):
    In,Out = [],[]
    for Dir in ResPaths:
        datapkl = "{}/Data.pkl".format(Dir)
        DataDict = ReadData(datapkl)

        paramfile = "{}/Parameters.py".format(Dir)
        Parameters = ReadParameters(paramfile)

        Coolant = [Parameters.Coolant[n] for n in ['Pressure','Temperature','Velocity']]
        In.append([*Parameters.CoilDisplacement,*Coolant,Parameters.Current])
        Out.append(DataDict['TC_Temp'].flatten())

    return In,Out

def Single(VL,DADict):
    Parameters = DADict['Parameters']
    DataFile_path = "{}/{}".format(VL.PROJECT_DIR, Parameters.DataFile)

    InputName = getattr(Parameters,'InputName','Input')
    OutputName = getattr(Parameters,'OutputName','Output')

    # Add data to data file
    _CompileData = getattr(Parameters,'_CompileData',None)
    if _CompileData:
        if type(_CompileData)==str:_CompileData=[_CompileData]
        for _data in _CompileData:
            ResDir = "{}/{}".format(VL.PROJECT_DIR,Parameters.TrainData)
            ResPaths = GetResPaths(ResDir)

            In, Out = CompileData(DADict,ResPaths)
            In, Out = np.array(In), np.array(Out)

            InPath = "{}/{}".format(_data,InputName)
            OutPath = "{}/{}".format(_data,OutputName)
            Writehdf(DataFile_path,In,InPath)
            Writehdf(DataFile_path,Out,OutPath)


    # Database = h5py.File(DataFile_path,'r')
    # TrainData,TestData = Parameters.TrainData, Parameters.TestData
    # TrainIn = Database["{}/{}".format(TrainData,InputName)]
    # TrainOut = Database["{}/{}".format(TrainData,OutputName)]
    # TestIn = Database["{}/{}".format(TestData,InputName)]
    # TestOut = Database["{}/{}".format(TestData,OutputName)]
    # Database.close()
