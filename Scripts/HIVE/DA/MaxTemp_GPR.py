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
        for resname in _CompileData:
            ResDir = "{}/{}".format(VL.PROJECT_DIR,resname)
            ResPaths = ML.GetResPaths(ResDir)

            In, Out = CompileData(DADict,ResPaths)
            In, Out = np.array(In), np.array(Out)

            InPath = "{}/{}".format(resname,InputName)
            OutPath = "{}/{}".format(resname,OutputName)
            ML.Writehdf(DataFile_path,In,InPath)
            ML.Writehdf(DataFile_path,Out,OutPath)

    Database = h5py.File(DataFile_path,'r')
    TrainData,TestData = Parameters.TrainData, Parameters.TestData
    TrainIn = Database["{}/{}".format(TrainData,InputName)][:]
    TrainOut = Database["{}/{}".format(TrainData,OutputName)][:].flatten()
    TestIn = Database["{}/{}".format(TestData,InputName)][:]
    TestOut = Database["{}/{}".format(TestData,OutputName)][:].flatten()
    Database.close()

    TrainIn,TrainOut = TrainIn.astype(dtype),TrainOut.astype(dtype)
    TestIn, TestOut = TestIn.astype(dtype), TestOut.astype(dtype)

    TrainIn = TrainIn[:Parameters.TrainNb]
    TrainOut = TrainOut[:Parameters.TrainNb]

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

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ML.ExactGPmodel(TrainIn_scale, TrainOut_scale, likelihood,
                Parameters.Kernel)

    ModelFile = '{}/Model.pth'.format(DADict["CALC_DIR"]) # File model will be saved to/loaded from
    if 0:
        lr = 0.01
        DADict['Data']['MSE'] = MSEvals = {}
        print()
        testdat = [TestIn_scale,TestOut_scale]
        testdat=[]
        Conv_P,MSE_P = model.Training(likelihood,3000,test=testdat,lr=lr,Print=100)
        print()
        print(MSE_P)

        torch.save(model.state_dict(), ModelFile)

        plt.figure()
        plt.plot(Conv_P)
        plt.savefig("{}/Convergence.eps".format(DADict["CALC_DIR"]),dpi=600)
        plt.close()

    else:
        state_dict = torch.load(ModelFile)
        model.load_state_dict(state_dict)

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

        if 0:
            UQ = TrainPred.stddev.numpy()*OutputScaler[1]
            sortix = np.argsort(UQ)[::-1]
            for pred,act,uq in zip(TrainPred_R[sortix],TrainOut[sortix],UQ[sortix]):
                print('Pred: {}, Act: {}, UQ: {}'.format(pred,act,uq))

        if 0:
            UQ = TestPred.stddev.numpy()*OutputScaler[1]
            sortix = np.argsort(UQ)[::-1]
            for pred,act,uq in zip(TestPred_R[sortix],TestOut[sortix],UQ[sortix]):
                print('Pred: {}, Act: {}, UQ: {}'.format(pred,act,uq))
