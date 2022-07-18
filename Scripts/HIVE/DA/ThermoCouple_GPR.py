import os
import h5py
import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt

from VLFunctions import ReadData, ReadParameters
from Optimise import FuncOpt
import ML

dtype = 'float64' # float64 is more accurate for optimisation purposes
torch_dtype = getattr(torch,dtype)
MaxCholeskySize = 1500

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
    TrainOut = Database["{}/{}".format(TrainData,OutputName)][:]
    TestIn = Database["{}/{}".format(TestData,InputName)][:]
    TestOut = Database["{}/{}".format(TestData,OutputName)][:]
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

    likelihoods, models = [], []
    for trainy in TrainOut_scale.T:
        _likelihood = gpytorch.likelihoods.GaussianLikelihood()
        _model = ML.ExactGPmodel(TrainIn_scale, trainy, _likelihood,
                    Parameters.Kernel)
        likelihoods.append(_likelihood)
        models.append(_model)

    model = gpytorch.models.IndependentModelList(*models)
    likelihood = gpytorch.likelihoods.LikelihoodList(*likelihoods)

    ModelFile = "{}/Model.pth".format(DADict['CALC_DIR'])
    if 0:
        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
        mll = gpytorch.mlls.SumMarginalLogLikelihood(likelihood, model)
        Iterations = 2000

        for i in range(Iterations):
            optimizer.zero_grad() # Zero gradients from previous iteration
            # Output from model
            with gpytorch.settings.max_cholesky_size(1500):
                output = model(*model.train_inputs)
                # Calc loss and backprop gradients
                loss = -mll(output, model.train_targets)
                loss.backward()
                optimizer.step()
                if i == 0 or i%10==0:
                    print(i, loss.item())

        torch.save(model.state_dict(), ModelFile)

        for mod in model.models:
            print('Lengthscale:',mod.covar_module.base_kernel.lengthscale.detach().numpy()[0])
            print('Outputscale', mod.covar_module.outputscale.detach().numpy())
            print('Noise',mod.likelihood.noise.detach().numpy()[0])
            print()

    else:
        state_dict = torch.load(ModelFile)
        model.load_state_dict(state_dict)

    model.eval()
    likelihood.eval()

    # x = np.ones((2,7))*0.5
    Initguess = np.random.uniform(0,1,size=(20,7))
    bnds = [[0,1]]*7
    Optima = FuncOpt(obj, Initguess, find='min', tol=0.00,
                     order='increasing',
                     bounds=bnds, jac=True, args=[model,TestOut_scale.detach().numpy()[0,:]])
    Coordinate, Value = Optima

    print('True y', TestOut_scale.detach().numpy()[0,:])
    x = TestIn_scale[0:1,:]
    pred = model(*[TestIn_scale[0:1,:]]*len(models))
    preds = [p.mean.detach().numpy() for p in pred]
    print('Pred y', preds)
    # x =
    print()
    print('True x', TestIn_scale.detach().numpy()[0,:])

    for c, v in zip(Coordinate, Value):
        print(v)
        print(c)
        print()
    # print(Value)

    # Pred, Grad = obj(x, model,)

    with torch.no_grad():
        # predictions = likelihood(*model(test_x, test_x))
        TrainPred = model(*[TrainIn_scale]*len(models))
        TestPred = model(*[TestIn_scale]*len(models))

        for i, (submodel, _TrainPred,_TestPred) in enumerate(zip(model.models, TrainPred, TestPred)):
            __TrainPred = ML.DataRescale(_TrainPred.mean.numpy(),*OutputScaler[:,i])
            __TestPred = ML.DataRescale(_TestPred.mean.numpy(),*OutputScaler[:,i])
            # print(__TrainPred[:5])
            # print(TrainOut[:5,i])
            TrainMSE = ((__TrainPred - TrainOut[:,i])**2)
            TestMSE = ((__TestPred - TestOut[:,i])**2)
            # if i==0:
            #     for j in range(300):
            #         print(__TestPred[j],'  ',TestOut[j,i],'  ',_TestPred.variance.numpy()[j]*OutputScaler[1,i]**2)

            print('TrainMSE',TrainMSE.mean())
            print('TestMSE',TestMSE.mean())


def obj(X, model, Target):
    X = torch.tensor(np.atleast_2d(X),dtype=torch_dtype)
    Preds, Grads = [], []
    for mod in model.models:
        _Pred = mod(X).mean.detach().numpy()
        _Grad = mod.Gradient(X).detach().numpy()


        Preds.append(_Pred)
        Grads.append(_Grad)
    Preds = np.array(Preds)
    # Grads = np.array(Grads)
    Grads = np.swapaxes(Grads,0,1)
    d = np.transpose(Preds - Target[:,None])
    Score = (d**2).sum(axis=1)
    dScore = (Grads*d[:,:,None]).sum(axis=1)

    return Score, dScore
