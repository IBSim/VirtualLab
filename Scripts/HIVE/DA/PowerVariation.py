import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from types import SimpleNamespace as Namespace
import torch
import gpytorch

from VLFunctions import ReadData, ReadParameters
import ML
from Optimise import FuncOpt
# from Sim.PreHIVE import ERMES

dtype = 'float64' # float64 is more accurate for optimisation purposes
torch_dtype = getattr(torch,dtype)

def MLMapping(Dir):
    datapkl = "{}/Data.pkl".format(Dir)
    DataDict = ReadData(datapkl)
    paramfile = "{}/Parameters.py".format(Dir)
    Parameters = ReadParameters(paramfile)

    In = [*Parameters.CoilDisplacement,Parameters.Rotation]
    Out = [DataDict['Power'],DataDict['Variation']]
    return In, Out

def ModelDefine(TrainIn,TrainOut,Kernel,prev_state=None):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ML.ExactGPmodel(TrainIn, TrainOut, likelihood, Kernel)
    if prev_state:
        state_dict = torch.load(prev_state)
        model.load_state_dict(state_dict)
    return likelihood, model

def Single(VL, DADict):
    Parameters = DADict["Parameters"]

    NbTorchThread = getattr(Parameters,'NbTorchThread',1)
    torch.set_default_dtype(torch_dtype)
    torch.set_num_threads(NbTorchThread)
    torch.manual_seed(getattr(Parameters,'Seed',100))

    DataFile_path = "{}/{}".format(VL.PROJECT_DIR,Parameters.DataFile)

    InputName = getattr(Parameters,'InputName','Input')
    OutputName = getattr(Parameters,'OutputName','Output')

    if hasattr(Parameters,'CompileData'):
        CompileData = Parameters.CompileData
        if type(CompileData)==str:CompileData = [CompileData]

        ResDirs = ["{}/{}".format(VL.PROJECT_DIR,resname) for resname in CompileData]
        InData, OutData = ML.CompileData(ResDirs,MLMapping)
        ML.WriteMLdata(DataFile_path, CompileData, InputName,
                       OutputName, InData, OutData)
