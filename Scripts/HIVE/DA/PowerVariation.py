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

def Single(VL, DADict):
    Parameters = DADict["Parameters"]

    torch.set_default_dtype(torch_dtype)
    # torch.random.manual_seed(200)
    # NbTorchThread = getattr(Parameters,'NbTorchThread',1)
    # torch.set_num_threads(NbTorchThread)

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

    TrainNb,TestNb = getattr(Parameters,'TrainNb',-1),getattr(Parameters,'TestNb',-1)
    TrainIn, TrainOut = ML.GetMLdata(DataFile_path, Parameters.TrainData,
                                    InputName, OutputName, TrainNb)
    TestIn, TestOut = ML.GetMLdata(DataFile_path, Parameters.TestData,
                                   InputName, OutputName, TestNb)

    NbInput,NbOutput = TrainIn.shape[1],TrainOut.shape[1]

    # Scale input to [0,1] (based on parameter space)
    # InputScaler = np.array([TrainIn.min(axis=0),TrainIn.max(axis=0) - TrainIn.min(axis=0)])
    DispX = DispY = [-0.01,0.01]
    DispZ,Rotation = [0.01,0.03],[-15,15]
    bounds = np.transpose([DispX,DispY,DispZ,Rotation]) # could import this for consistency
    InputScaler = np.array([bounds[0],bounds[1] - bounds[0]])
    TrainIn_scale = ML.DataScale(TrainIn,*InputScaler)
    TestIn_scale = ML.DataScale(TestIn,*InputScaler)

    # Scale output to [0,1]
    OutputScaler = np.array([TrainOut.min(axis=0),TrainOut.max(axis=0) - TrainOut.min(axis=0)])
    TrainOut_scale = ML.DataScale(TrainOut,*OutputScaler)
    TestOut_scale = ML.DataScale(TestOut,*OutputScaler)

    # Convert to tensors
    TrainIn_scale = torch.from_numpy(TrainIn_scale)
    TrainOut_scale = torch.from_numpy(TrainOut_scale)
    TestIn_scale = torch.from_numpy(TestIn_scale)
    TestOut_scale = torch.from_numpy(TestOut_scale)

    ModelFile = "{}/Model.pth".format(DADict['CALC_DIR'])
    if Parameters.Train:
        min_noise = getattr(Parameters,'MinNoise',None)
        likelihood, model = ML.GPRModel_Multi(TrainIn_scale, TrainOut_scale,
                                        Parameters.Kernel,min_noise=min_noise)

        ML.GPR_Train_Multi(model, Parameters.Epochs)

        torch.save(model.state_dict(), ModelFile)

        # for mod in model.models:
        #     print('Lengthscale:',mod.covar_module.base_kernel.lengthscale.detach().numpy()[0])
        #     print('Outputscale', mod.covar_module.outputscale.detach().numpy())
        #     print('Noise',mod.likelihood.noise.detach().numpy()[0])
        #     print()

        # TrainMSE, TestMSE = np.array(TrainMSE),np.array(TestMSE)
        # plt.figure()
        # l = 2
        # plt.plot(TrainMSE[l:,0],TrainMSE[l:,1],label='Power')
        # plt.plot(TrainMSE[l:,0],TrainMSE[l:,2],label='Variation')
        # plt.legend()
        # plt.show()
        #

    else:
        likelihood, model = ML.GPRModel_Multi(TrainIn_scale,TrainOut_scale,
                                        Parameters.Kernel,prev_state=ModelFile)

    model.eval();likelihood.eval()

    with torch.no_grad():
        TrainPred = model(*[TrainIn_scale]*NbOutput)
        TestPred = model(*[TestIn_scale]*NbOutput)

        for i in range(NbOutput):
            Train_mean = TrainPred[i].mean.numpy()
            Test_mean = TestPred[i].mean.numpy()

            TrainMSE = ML.MSE(Train_mean,TrainOut_scale[:,i].numpy())
            TestMSE = ML.MSE(Test_mean,TestOut_scale[:,i].numpy())
            print('Train_scale',TrainMSE)
            print('Test_scale',TestMSE)

            TrainPred_R = ML.DataRescale(Train_mean,*OutputScaler[:,i])
            TestPred_R = ML.DataRescale(Test_mean,*OutputScaler[:,i])
            TrainMSE_R = ML.MSE(TrainPred_R,TrainOut[:,i])
            TestMSE_R = ML.MSE(TestPred_R,TestOut[:,i])
            print('Train',TrainMSE_R)
            print('Test',TestMSE_R)
            print()

    Adaptive = getattr(Parameters,'Adaptive',{})
    if Adaptive:
        Method = Adaptive['Method']
        NbNext = Adaptive['Nb']
        NbCand = Adaptive['NbCandidates']
        Seed = Adaptive.get('Seed',None)
        bndmax = Adaptive.get('bndmax',0)

        if Seed!=None: np.random.seed(Seed)
        Candidates = np.random.uniform(0,1,size=(NbCand,NbInput))

        sort=True
        BestPoints = []
        for i in range(NbNext):
            if Method.lower()=='ei':
                score,srtCandidates = ML.EI_Multi(model,Candidates,sort=sort)
            if Method.lower()=='eigf':
                score,srtCandidates = ML.EIGF_Multi(model,Candidates,sort=sort)
            if Method.lower()=='maxei':
                score,srtCandidates = ML.MaxEI_Multi(model,Candidates,sort=sort)
            if Method.lower()=='eigrad':
                score,srtCandidates = ML.EIGrad_Multi(model,Candidates,sort=sort)

            # for _s, _c in zip(score[:5], srtCandidates):
            #     print(_c,_s)
            # print()

            BestPoint = srtCandidates[0:1]
            BestPoint_pth = torch.from_numpy(BestPoint)
            with torch.no_grad():
                output = model(*[BestPoint_pth]*NbOutput)
            for j,mod in enumerate(model.models):
                _mod = mod.get_fantasy_model(BestPoint_pth,output[j].mean)
                model.models[j] = _mod

            Candidates = srtCandidates[1:]
            BestPoints.append(BestPoint.flatten())
        print(np.around(BestPoints,3))
        BestPoints = ML.DataRescale(np.array(BestPoints),*InputScaler)
        # print(BestPoints)
        DADict['Data']['BestPoints'] = BestPoints
