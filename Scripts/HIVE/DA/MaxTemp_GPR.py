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

DispX = DispY = [-0.01,0.01]
DispZ = [0.01,0.02]
CoolantP, CoolantT, CoolantV = [0.4,1.6], [30,70], [5,15]
Current = [600,1000]

InTag = ['Coil X','Coil Y','Coil Z', 'Coolant Pressure',
         'Coolant Temperature', 'Coolant Velocity','Coil Current']
OutTag = ['MaxTemperature','MaxStress']

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
    Out = [DataDict['MaxTemp'], DataDict['MaxStress']]
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

    # TrainOut, TestOut = TrainOut[:,0:1], TestOut[:,0:1]
    NbInput,NbOutput = TrainIn.shape[1],TrainOut.shape[1]

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
        prev_state = getattr(Parameters,'PrevState',None)
        if prev_state==True: prev_state = ModelFile
        likelihood, model = ML.Create_GPR(TrainIn_scale, TrainOut_scale, Parameters.Kernel,
                                          prev_state=prev_state, min_noise=min_noise)

        # Train model
        TrainDict = getattr(Parameters,'TrainDict',{})
        Conv = ML.GPR_Train(model, **TrainDict)

        # Save model
        torch.save(model.state_dict(), ModelFile)

        # Plot convergence & save
        plt.figure()
        for j, _Conv in enumerate(Conv):
            plt.plot(_Conv,label='Output_{}'.format(j))
        plt.legend()
        plt.savefig("{}/Convergence.eps".format(DADict["CALC_DIR"]),dpi=600)
        plt.close()
    else:
        # Load previously trained model
        likelihood, model = ML.Create_GPR(TrainIn_scale, TrainOut_scale,
                                        Parameters.Kernel, prev_state=ModelFile)
    model.eval(); likelihood.eval()
    
    # =========================================================================
    # Get error metrics for model
    df_train = ML.GetMetrics(model,TrainIn_scale,TrainOut_scale.detach().numpy())
    df_test = ML.GetMetrics(model,TestIn_scale,TestOut_scale.detach().numpy())
    print('\nTrain metrics')
    print(df_train)
    print('\nTest metrics')
    print(df_test)
    print()

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
    for i, mod in enumerate(model.models):
        val,cd = ML.GetExtrema(mod, 100, bounds)
        cd = ML.DataRescale(cd,*InputScaler)
        val = ML.DataRescale(val,*OutputScaler[:,i])
        RangeDict["Min_{}".format(OutTag[i])] = val[0]
        RangeDict["Max_{}".format(OutTag[i])] = val[1]
        print("{}\nMin:{}, Max:{}\n".format(OutTag[i],*np.around(val,2)))

    # ==========================================================================
    # See impact of varying inputs
    if False:
        n=10
        split = np.linspace(0,1,n+1)
        a = [0.5]*NbInput
        Res = []
        for i, var in enumerate(InTag):
            n = []
            for p in split:
                _a = a.copy()
                _a[i] = p
                n.append(_a)

            out = []
            for mod in model.models:
                with torch.no_grad():
                    n = torch.tensor(n)
                    _out = mod(n).mean.numpy()
                out.append(_out)
            out = np.array(out).T
            out = ML.DataRescale(out,*OutputScaler)
            Res.append(out)

        # Plot individually
        for i, (tag,res) in enumerate(zip(InTag,Res)):
            fig,(ax1,ax2) = plt.subplots(2,figsize=(10,10),sharex=True)
            fig.suptitle(tag.capitalize())
            _split = InputScaler[0,i] + split*InputScaler[1,i]
            ax1.plot(_split,res[:,0])
            ax1.set_ylabel('Temperature')
            ax2.plot(_split,res[:,1])
            ax2.set_ylabel('Stress')
            plt.show()

        # Plot together to compare impact
        fig,(ax1,ax2) = plt.subplots(2,figsize=(10,10))
        for tag,res in zip(InTag,Res):
            ax1.plot(split,res[:,0],label=tag.capitalize())
            ax2.plot(split,res[:,1],label=tag.capitalize())
        ax1.set_ylabel('Temperature')
        ax2.set_ylabel('Stress')
        ax1.legend()
        ax2.legend()
        plt.show()

    # ==========================================================================
    # See range for stressing a component for a required temperature.
    if False:
        space=200
        rdlow = int(np.ceil(RangeDict['Min_MaxTemperature'] / space)) * space
        rdhigh = int(np.ceil(RangeDict['Max_MaxTemperature'] / space)) * space
        ExactTemps = list(range(rdlow,rdhigh,space))
        MinStresses, MaxStresses = [], []
        for ExactTemp in ExactTemps:
            ExactTempScale = ML.DataScale(ExactTemp,*OutputScaler[:,0])
            con = ML.FixedBound(model.models[0],ExactTempScale)

            Opt_cd_min, Opt_val_min = ML.GetOptima(model.models[1], 100, bounds,
                                           find='min',constraints=con,maxiter=30)
            Opt_cd_max, Opt_val_max = ML.GetOptima(model.models[1], 100, bounds,
                                           find='max',constraints=con,maxiter=30)

            MinStress = ML.DataRescale(Opt_val_min[0],*OutputScaler[:,1])
            MaxStress = ML.DataRescale(Opt_val_max[0],*OutputScaler[:,1])

            MinStresses.append(MinStress)
            MaxStresses.append(MaxStress)

            print("Max. Component Temperature: {}".format(ExactTemp))
            print("Stress Min.: {:.2f}, Stress Max.: {:.2f}\n".format(MinStress,MaxStress))

            # _Opt_cd_min = ML.DataRescale(Opt_cd_min,*InputScaler)
            # Opt_cd_min = torch.from_numpy(Opt_cd_min)
            # with torch.no_grad():
            #     for i,mod in enumerate(model.models):
            #         pred = mod(Opt_cd_min).mean.numpy()
            #         preds = ML.DataRescale(pred,*OutputScaler[:,i])
            #         print(preds)
            # print()

        plt.figure()
        plt.plot(ExactTemps,MinStresses,label='Min')
        plt.plot(ExactTemps,MaxStresses,label='Max')
        plt.xlabel('Component Max. Temperature (C)')
        plt.ylabel('Component Max. Stress (MPa)')
        plt.legend()
        plt.show()

    if False:
        for j,mod in enumerate(model.models):
            with torch.no_grad():
                TrainPred = mod(TrainIn_scale)
                TestPred = mod(TestIn_scale)

            TrainPred_R = ML.DataRescale(TrainPred.mean.numpy(),*OutputScaler[:,j])
            TestPred_R = ML.DataRescale(TestPred.mean.numpy(),*OutputScaler[:,j])
            if 0:
                UQ = TrainPred.stddev.numpy()*OutputScaler[1,j]
                sortix = np.argsort(UQ)[::-1]
                for pred,act,uq in zip(TrainPred_R[sortix],TrainOut[sortix],UQ[sortix]):
                    print('Pred: {}, Act: {}, UQ: {}'.format(pred,act,uq))
            if 1:
                UQ = TestPred.stddev.numpy()*OutputScaler[1,j]
                sortix = np.argsort(UQ)[::-1]
                for pred,act,uq in zip(TestPred_R[sortix],TestOut[sortix,j],UQ[sortix]):
                    print('Pred: {}, Act: {}, UQ: {}'.format(pred,act,uq))
                print()

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
        likelihood, model = ML.Create_GPR(TrainIn_scale,TrainOut_scale,
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
