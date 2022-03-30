import os
import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
import time

from VLFunctions import ReadData, ReadParameters
from Scripts.Common.ML import ML, Adaptive

dtype = 'float64' # float64 is more accurate for optimisation purposes
torch_dtype = getattr(torch,dtype)
torch.set_default_dtype(torch_dtype)

InTag = ['x','y','z','pressure','temperature','velocity','current']
DispX = DispY = [-0.01,0.01]
DispZ = [0.01,0.02]
CoolantP, CoolantT, CoolantV = [0.4,1.6], [30,70], [5,15]
Current = [600,1000]

bounds = np.transpose([DispX,DispY,DispZ,CoolantP,CoolantT,CoolantV,Current])
InputScaler = np.array([bounds[0],bounds[1] - bounds[0]])

# ==============================================================================
# Functions for gathering necessary data and writing to file
def MLMapping(Dir):
    datapkl = "{}/Data.pkl".format(Dir)
    DataDict = ReadData(datapkl)
    paramfile = "{}/Parameters.py".format(Dir)
    Parameters = ReadParameters(paramfile)

    Coolant = [Parameters.Coolant[n] for n in ['Pressure','Temperature','Velocity']]
    In = [*Parameters.CoilDisplacement,*Coolant,Parameters.Current]
    Out = DataDict['MaxTemp']
    return In, Out

def CompileData(VL,DADict):
    Parameters = DADict["Parameters"]

    DataFile_path = "{}/{}".format(VL.PROJECT_DIR,Parameters.DataFile)
    InputName = getattr(Parameters,'InputName','Input')
    OutputName = getattr(Parameters,'OutputName','Output')

    CmpData = Parameters.CompileData
    if type(CmpData)==str:CmpData = [CmpData]

    ResDirs = ["{}/{}".format(VL.PROJECT_DIR,resname) for resname in CmpData]
    InData, OutData = ML.CompileData(ResDirs,MLMapping)
    ML.WriteMLdata(DataFile_path, CmpData, InputName,
                   OutputName, InData, OutData)

# ==============================================================================
# default function
def Single(VL,DADict):
    Parameters = DADict['Parameters']

    # ==========================================================================
    # Compile data here if needed
    if getattr(Parameters,'CompileData',None) !=None:
        CompileData(VL,DADict)

    # ==========================================================================
    # Get Train & test data and scale
    DataFile_path = "{}/{}".format(VL.PROJECT_DIR,Parameters.DataFile)
    InputName = getattr(Parameters,'InputName','Input')
    OutputName = getattr(Parameters,'OutputName','Output')

    TrainIn, TrainOut = ML.GetMLdata(DataFile_path, Parameters.TrainData, InputName,
                                     OutputName,getattr(Parameters,'TrainNb',-1))
    TestIn, TestOut = ML.GetMLdata(DataFile_path, Parameters.TestData, InputName,
                                   OutputName,getattr(Parameters,'TestNb',-1))

    NbInput,NbOutput = TrainIn.shape[1],TrainOut.shape[1]

    TrainOut, TestOut = TrainOut.flatten(), TestOut.flatten()

    # Scale input to [0,1] (based on parameter space)
    TrainIn_scale = ML.DataScale(TrainIn,*InputScaler)
    TestIn_scale = ML.DataScale(TestIn,*InputScaler)

    # Scale output to [0,1]
    OutputScaler = np.array([TrainOut.min(axis=0),TrainOut.max(axis=0) - TrainOut.min(axis=0)])
    TrainOut_scale = ML.DataScale(TrainOut,*OutputScaler)
    TestOut_scale = ML.DataScale(TestOut,*OutputScaler)

    TrainIn_scale = torch.from_numpy(TrainIn_scale)
    TrainOut_scale = torch.from_numpy(TrainOut_scale)
    TestIn_scale = torch.from_numpy(TestIn_scale)
    TestOut_scale = torch.from_numpy(TestOut_scale)

    # ==========================================================================
    # Train a new model or load an old one
    ModelFile = '{}/Model.pth'.format(DADict["CALC_DIR"]) # Saved model location
    if Parameters.Train:
        # get model & likelihoods
        min_noise = getattr(Parameters,'MinNoise',None)
        likelihood, model = ML.GPRModel_Single(TrainIn_scale, TrainOut_scale,
                                        Parameters.Kernel, min_noise=min_noise)
        # Train model
        testdat = [] #[TestIn_scale,TestOut_scale]
        Conv_P = model.Training(likelihood,Parameters.Epochs,test=testdat, Print=100)[0]
        # Save model
        torch.save(model.state_dict(), ModelFile)
        # Plot convergence & save
        plt.figure()
        plt.plot(Conv_P)
        plt.savefig("{}/Convergence.eps".format(DADict["CALC_DIR"]),dpi=600)
        plt.close()

        # Print model parameters
        print('Lengthscale:',model.covar_module.base_kernel.lengthscale.detach().numpy()[0])
        print('Outputscale', model.covar_module.outputscale.detach().numpy())
        print('Noise',model.likelihood.noise.detach().numpy()[0])
    else:
        # Load previously trained model
        likelihood, model = ML.GPRModel_Single(TrainIn_scale, TrainOut_scale,
                                        Parameters.Kernel, ModelFile)
    model.eval(); likelihood.eval()

    # =========================================================================
    # Get error metrics for model
    TrainMetrics = ML.GetMetrics(model, TrainIn_scale, TrainOut_scale.detach().numpy())
    TestMetrics = ML.GetMetrics(model, TestIn_scale, TestOut_scale.detach().numpy())

    for tp,data in zip(['Train','Test'],[TrainMetrics,TestMetrics]):
        outstr = "{} Data\n".format(tp)
        for i,metric in enumerate(['MSE','MAE','RMSE','R^2']):
            outstr+="{}: {}\n".format(metric,data[i])
        print(outstr)

    # ==========================================================================
    # Get next points to collect data
    bounds = [[0,1]]*NbInput
    AdaptDict = getattr(Parameters,'Adaptive',{})
    if AdaptDict:
        BestPoints = Adaptive.Adaptive(model, AdaptDict, bounds)
        print(np.around(BestPoints,3))
        BestPoints = ML.DataRescale(np.array(BestPoints),*InputScaler)
        DADict['Data']['BestPoints'] = BestPoints


    # ==========================================================================
    # Get min and max values for each
    print('Extrema')
    RangeDict = {}
    val,cd = ML.GetExtrema(model, 50, bounds)
    cd = ML.DataRescale(cd,*InputScaler)
    val = ML.DataRescale(val,*OutputScaler)
    RangeDict["Min"] = val[0]
    RangeDict["Max"] = val[1]
    print("Min:{}, Max:{}\n".format(*np.around(val,2)))


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
        if 1:
            UQ = TestPred.stddev.numpy()*OutputScaler[1]
            sortix = np.argsort(UQ)[::-1]
            sortix = np.argsort(TestMSE)[::-1]
            for pred,act,uq in zip(TestPred_R[sortix],TestOut[sortix],UQ[sortix]):
                print('Pred: {}, Act: {}, UQ: {}'.format(pred,act,uq))


# ==============================================================================
# Build a committee of models for query by committee adaptive scheme
def CommitteeBuild(VL,DADict):
    Parameters = DADict['Parameters']

    Likelihoods, Models = [], []
    for modeldir in Parameters.CommitteeModels:
        dirfull = "{}/{}".format(VL.PROJECT_DIR,modeldir)
        paramfile = "{}/Parameters.py".format(dirfull)
        ModParameters = ReadParameters(paramfile)
        DataFile_path = "{}/{}".format(VL.PROJECT_DIR,ModParameters.DataFile)
        InputName = getattr(Parameters,'InputName','Input')
        OutputName = getattr(Parameters,'OutputName','Output')
        TrainIn, TrainOut = ML.GetMLdata(DataFile_path, ModParameters.TrainData,
                                        InputName, OutputName, ModParameters.TrainNb)

        TrainOut = TrainOut.flatten()
        TrainIn_scale = ML.DataScale(TrainIn,*InputScaler)
        TrainIn_scale = torch.from_numpy(TrainIn_scale)

        OutputScaler = np.array([TrainOut.min(axis=0),TrainOut.max(axis=0) - TrainOut.min(axis=0)])
        TrainOut_scale = ML.DataScale(TrainOut,*OutputScaler)
        TrainOut_scale = torch.from_numpy(TrainOut_scale)

        ModelFile = "{}/Model.pth".format(dirfull)
        likelihood, model = ML.GPRModel_Single(TrainIn_scale,TrainOut_scale,
                                        ModParameters.Kernel,prev_state=ModelFile)

        likelihood.eval();model.eval()
        Likelihoods.append(likelihood)
        Models.append(model)

    bounds = [[0,1]]*len(InTag)
    AdaptDict = getattr(Parameters,'Adaptive',{})
    if AdaptDict:
        st = time.time()
        BestPoints = Adaptive.Adaptive(Models, AdaptDict, bounds,Show=3)
        end = time.time() - st
        print('Time',end)
        print(np.around(BestPoints,3))
        BestPoints = ML.DataRescale(np.array(BestPoints),*InputScaler)
        DADict['Data']['BestPoints'] = BestPoints











#
