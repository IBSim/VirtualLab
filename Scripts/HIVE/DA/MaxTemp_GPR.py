import os
import h5py
import numpy as np
from natsort import natsorted
import pickle
import torch
import gpytorch
import matplotlib.pyplot as plt

from VLFunctions import ReadData, ReadParameters
from Optimise import FuncOpt
import ML

dtype = 'float64' # float64 is more accurate for optimisation purposes
torch_dtype = getattr(torch,dtype)

def MLMapping(Dir):
    datapkl = "{}/Data.pkl".format(Dir)
    DataDict = ReadData(datapkl)
    paramfile = "{}/Parameters.py".format(Dir)
    Parameters = ReadParameters(paramfile)

    Coolant = [Parameters.Coolant[n] for n in ['Pressure','Temperature','Velocity']]
    In = [*Parameters.CoilDisplacement,*Coolant,Parameters.Current]
    Out = DataDict['MaxTemp']
    return In, Out

def ModelDefine(TrainIn,TrainOut,Kernel,prev_state=None):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ML.ExactGPmodel(TrainIn, TrainOut, likelihood, Kernel)
    if prev_state:
        state_dict = torch.load(prev_state)
        model.load_state_dict(state_dict)
    return likelihood, model

def Single(VL,DADict):
    Parameters = DADict['Parameters']
    DataFile_path = "{}/{}".format(VL.PROJECT_DIR,Parameters.DataFile)

    InputName = getattr(Parameters,'InputName','Input')
    OutputName = getattr(Parameters,'OutputName','Output')

    if hasattr(Parameters,'CompileData'):
        _CompileData = Parameters.CompileData
        if type(_CompileData)==str:_CompileData = [CompileData]

        ResDirs = ["{}/{}".format(VL.PROJECT_DIR,resname) for resname in _CompileData]
        InData, OutData = ML.CompileData(ResDirs,MLMapping)#
        ML.WriteMLdata(DataFile_path, _CompileData, InputName,
                       OutputName, InData, OutData)

    TrainIn, TrainOut = ML.GetMLdata(DataFile_path, Parameters.TrainData, InputName,
                                     OutputName,getattr(Parameters,'TrainNb',-1))

    TestIn, TestOut = ML.GetMLdata(DataFile_path, Parameters.TestData, InputName,
                                   OutputName,getattr(Parameters,'TestNb',-1))

    TrainOut, TestOut = TrainOut.flatten(), TestOut.flatten()

    NbInput = TrainIn.shape[1] if TrainIn.ndim >1 else 1
    NbOutput = TrainOut.shape[1] if TrainOut.ndim>1 else 1

    TrainIn,TrainOut = TrainIn.astype(dtype),TrainOut.astype(dtype)
    TestIn, TestOut = TestIn.astype(dtype), TestOut.astype(dtype)

    # Scale test & train input * output data to [0,1] (based on training data)
    InputScaler = np.array([TrainIn.min(axis=0),TrainIn.max(axis=0) - TrainIn.min(axis=0)])
    OutputScaler = np.array([TrainOut.min(axis=0),TrainOut.max(axis=0) - TrainOut.min(axis=0)])

    TrainIn_scale = ML.DataScale(TrainIn,*InputScaler)
    TrainOut_scale = ML.DataScale(TrainOut,*OutputScaler)
    TestIn_scale = ML.DataScale(TestIn,*InputScaler)
    TestOut_scale = ML.DataScale(TestOut,*OutputScaler)

    TrainIn_scale = torch.from_numpy(TrainIn_scale)
    TrainOut_scale = torch.from_numpy(TrainOut_scale)
    TestIn_scale = torch.from_numpy(TestIn_scale)
    TestOut_scale = torch.from_numpy(TestOut_scale)

    ModelFile = '{}/Model.pth'.format(DADict["CALC_DIR"]) # File model will be saved to/loaded from
    if Parameters.Train:
        likelihood, model = ModelDefine(TrainIn_scale, TrainOut_scale, Parameters.Kernel)
        lr = 0.01
        DADict['Data']['MSE'] = MSEvals = {}
        print()
        testdat = [TestIn_scale,TestOut_scale]
        testdat=[]
        Conv_P,MSE_P = model.Training(likelihood,Parameters.Epochs,test=testdat,lr=lr,Print=100)
        print()

        torch.save(model.state_dict(), ModelFile)

        plt.figure()
        plt.plot(Conv_P)
        plt.savefig("{}/Convergence.eps".format(DADict["CALC_DIR"]),dpi=600)
        plt.close()
    else:
        likelihood, model = ModelDefine(TrainIn_scale, TrainOut_scale,
                                        Parameters.Kernel, ModelFile)

    model.eval(); likelihood.eval()

    with torch.no_grad():
        TrainPred = model(TrainIn_scale)
        TestPred = model(TestIn_scale)

        TrainPred_R = ML.DataRescale(TrainPred.mean.numpy(),*OutputScaler)
        TestPred_R = ML.DataRescale(TestPred.mean.numpy(),*OutputScaler)
        TrainMSE = ((TrainPred_R - TrainOut)**2)
        TestMSE = ((TestPred_R - TestOut)**2)

        print(TrainMSE.mean())
        print(TestMSE.mean())

        # if 0:
        #     UQ = TrainPred.stddev.numpy()*OutputScaler[1]
        #     sortix = np.argsort(UQ)[::-1]
        #     for pred,act,uq in zip(TrainPred_R[sortix],TrainOut[sortix],UQ[sortix]):
        #         print('Pred: {}, Act: {}, UQ: {}'.format(pred,act,uq))
        #
        # if 0:
        #     UQ = TestPred.stddev.numpy()*OutputScaler[1]
        #     sortix = np.argsort(UQ)[::-1]
        #     for pred,act,uq in zip(TestPred_R[sortix],TestOut[sortix],UQ[sortix]):
        #         print('Pred: {}, Act: {}, UQ: {}'.format(pred,act,uq))














#
