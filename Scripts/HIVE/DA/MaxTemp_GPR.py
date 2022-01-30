import os
import h5py
import numpy as np
from natsort import natsorted
import pickle
import torch
import gpytorch
import matplotlib.pyplot as plt

from Functions import DataScale, DataRescale, GetResPaths, ReadData, ReadParameters, Writehdf
from Optimise import FuncOpt
from MLModels import ExactGPmodel

dtype = 'float64' # float64 is more accurate for optimisation purposes
torch_dtype = getattr(torch,dtype)

def CompileData(DADict,Dirs):
    In,Out = [],[]
    for Dir in Dirs:
        datapkl = "{}/Data.pkl".format(Dir)
        with open(datapkl,'rb') as f:
            Data = pickle.load(f)
            Out.append(Data['MaxTemp'])
        parapkl = "{}/.Parameters.pkl".format(Dir)
        with open(parapkl,'rb') as f:
            Para = pickle.load(f)
            Coolant = [Para.Coolant[n] for n in ['Pressure','Temperature','Velocity']]
            In.append([*Para.CoilDisplacement,*Coolant,Para.Current])

    return In,Out

def Single(VL,DADict):
    Parameters = DADict['Parameters']
    DataFile_path = "{}/{}".format(VL.PROJECT_DIR,Parameters.DataFile)

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
    # Input_Train = Database["{}/{}".format(TrainData,InputName)]
    # Output_Train = Database["{}/{}".format(TrainData,OutputName)]
    # Input_Test = Database["{}/{}".format(TestData,InputName)]
    # Output_Test = Database["{}/{}".format(TestData,OutputName)]
    # Database.close()
