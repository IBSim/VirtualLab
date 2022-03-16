import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from types import SimpleNamespace as Namespace
import torch
import gpytorch

from VLFunctions import ReadData, ReadParameters
from Scripts.Common.ML import ML, Adaptive

dtype = 'float64' # float64 is more accurate for optimisation purposes
torch_dtype = getattr(torch,dtype)

InTag, OutTag = ['x','y','z','rotation'],['Power','Variation']

DispX = DispY = [-0.01,0.01]
DispZ,Rotation = [0.01,0.03],[-15,15]
bounds = np.transpose([DispX,DispY,DispZ,Rotation]) # could import this for consistency
InputScaler = np.array([bounds[0],bounds[1] - bounds[0]])

# ==============================================================================
# Functions for gathering necessary data and writing to file
def MLMapping(Dir):
    datapkl = "{}/Data.pkl".format(Dir)
    DataDict = ReadData(datapkl)
    paramfile = "{}/Parameters.py".format(Dir)
    Parameters = ReadParameters(paramfile)

    In = [*Parameters.CoilDisplacement,Parameters.Rotation]
    Out = [DataDict['Power'],DataDict['Variation']]
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
def Single(VL, DADict):
    Parameters = DADict["Parameters"]

    torch.set_default_dtype(torch_dtype)
    # torch.random.manual_seed(200)
    # NbTorchThread = getattr(Parameters,'NbTorchThread',1)
    # torch.set_num_threads(NbTorchThread)

    DataFile_path = "{}/{}".format(VL.PROJECT_DIR,Parameters.DataFile)
    InputName = getattr(Parameters,'InputName','Input')
    OutputName = getattr(Parameters,'OutputName','Output')

    if getattr(Parameters,'CompileData',None):
        CompileData(VL,DADict)

    # ==========================================================================
    # Get Train & test data and scale
    TrainNb, TestNb = getattr(Parameters,'TrainNb',-1),getattr(Parameters,'TestNb',-1)
    TrainIn, TrainOut = ML.GetMLdata(DataFile_path, Parameters.TrainData,
                                    InputName, OutputName, TrainNb)
    TestIn, TestOut = ML.GetMLdata(DataFile_path, Parameters.TestData,
                                   InputName, OutputName, TestNb)

    NbInput,NbOutput = TrainIn.shape[1],TrainOut.shape[1]

    # Scale input to [0,1] (based on parameter space)
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

    # ==========================================================================
    # Train a new model or load an old one
    ModelFile = "{}/Model.pth".format(DADict['CALC_DIR']) # Saved model location
    if Parameters.Train:
        # get model & likelihoods
        min_noise = getattr(Parameters,'MinNoise',None)
        prev_state = getattr(Parameters,'PrevState',None)
        if prev_state==True: prev_state = ModelFile

        likelihood, model = ML.Create_GPR(TrainIn_scale, TrainOut_scale,
                                        Parameters.Kernel,min_noise=min_noise)

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
        likelihood, model = ML.Create_GPR(TrainIn_scale,TrainOut_scale,
                                        Parameters.Kernel,prev_state=ModelFile)
    model.eval();likelihood.eval()

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
        val,cd = ML.GetExtrema(mod, 50, bounds)
        cd = ML.DataRescale(cd,*InputScaler)
        val = ML.DataRescale(val,*OutputScaler[:,i])
        RangeDict["Min_{}".format(OutTag[i])] = val[0]
        RangeDict["Max_{}".format(OutTag[i])] = val[1]
        print("{}\nMin:{}, Max:{}\n".format(OutTag[i],*np.around(val,2)))

    # ==========================================================================
    # Get minimum variation for different powers & plot
    if True:
        space = 100
        rdlow = int(np.ceil(RangeDict['Min_Power'] / space)) * space
        rdhigh = int(np.ceil(RangeDict['Max_Power'] / space)) * space
        P,V = [],[]
        for i in range(rdlow,rdhigh,space):
            iscale = ML.DataScale(i,*OutputScaler[:,0])
            con = ML.LowerBound(model.models[0], iscale)
            Opt_cd, Opt_val = ML.GetOptima(model.models[1], 100, bounds,
                                           find='min', constraints=con, maxiter=30)

            Opt_cd = ML.DataRescale(Opt_cd,*InputScaler)
            Opt_val = ML.DataRescale(Opt_val,*OutputScaler[:,1])
            mess = 'Minimised variaition for power above {} W:\n{}, {}\n'.format(i,Opt_cd[0],Opt_val[0])
            print(mess)
            P.append(i);V.append(Opt_val[0])

        plt.figure()
        plt.xlabel('Power')
        plt.ylabel('Variation')
        plt.plot(P,V)
        plt.show()


    '''
    AxMaj,ResAx = [2,3],0
    grid = Gridmaker(AxMaj,ResAx)
    grid_unroll = grid.reshape((int(grid.size/NbInput),NbInput))
    grid_unroll = torch.tensor(grid_unroll, dtype=torch_dtype)
    with torch.no_grad():
        out_grid = model.models[ResAx](grid_unroll).mean.numpy()
    out_grid = ML.DataRescale(out_grid,*OutputScaler[:,ResAx])
    outmax,outmin = out_grid.max(),out_grid.min()
    out_grid = out_grid.reshape(grid.shape[:-1])

    fig, ax = plt.subplots(nrows=7, ncols=7, sharex=True, sharey=True, dpi=200, figsize=(12,9))
    ax = np.atleast_2d(ax)
    fig.subplots_adjust(right=0.8)
    for i in range(7):
        _i = -(i+1)
        for j in range(7):
            sl = [slice(None)]*NbInput
            sl[AxMaj[0]],sl[AxMaj[1]]  = i,j
            Im = ax[_i,j].imshow(out_grid[tuple(sl)].T, cmap = 'coolwarm', norm=LogNorm(vmax=outmax, vmin=outmin),
                                                        origin='lower',extent=(0,1,0,1))
    plt.show()
    '''

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
        TrainIn, TrainOut = ML.GetMLdata(DataFile_path, ModParameters.TrainData,
                                        'Input', 'Output', ModParameters.TrainNb)
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
        BestPoints = Adaptive.Adaptive(Models, AdaptDict, bounds,Show=3)
        print(np.around(BestPoints,3))
        BestPoints = ML.DataRescale(np.array(BestPoints),*InputScaler)
        DADict['Data']['BestPoints'] = BestPoints

# ==============================================================================
def Gridmaker(AxMaj,ResAx,MajorN=7,MinorN=20):
    AxMin = list(set(range(len(InTag))).difference(AxMaj))
    # Discretisation
    DiscMin = np.linspace(0+0.5*1/MinorN,1-0.5*1/MinorN,MinorN)
    DiscMaj = np.linspace(0,1,MajorN)
    # DiscMaj = np.linspace(0+0.5*1/MajorN,1-0.5*1/MajorN,MajorN)
    disc = [DiscMaj]*4
    disc[AxMin[0]] = disc[AxMin[1]] = DiscMin
    grid = np.meshgrid(*disc, indexing='ij')
    grid = np.moveaxis(np.array(grid),0,-1) #grid point is now the last axis
    return grid
# ==============================================================================

def GetModel(resdir,DataFile_path):
    paramfile = "{}/Parameters.py".format(resdir)
    Parameters = ReadParameters(paramfile)

    TrainIn, TrainOut = ML.GetMLdata(DataFile_path, Parameters.TrainData,
                                    'Input', 'Output', Parameters.TrainNb)
    TrainIn_scale = ML.DataScale(TrainIn,*InputScaler)
    TrainIn_scale = torch.from_numpy(TrainIn_scale)

    OutputScaler = np.array([TrainOut.min(axis=0),TrainOut.max(axis=0) - TrainOut.min(axis=0)])
    TrainOut_scale = ML.DataScale(TrainOut,*OutputScaler)
    TrainOut_scale = torch.from_numpy(TrainOut_scale)

    # print(mldir,len(TrainOut),TrainOut.max(axis=0))
    NbInput,NbOutput = TrainIn.shape[1],TrainOut.shape[1]
    TrainNb = len(TrainIn)

    ModelFile = "{}/Model.pth".format(resdir)
    likelihood, model = ML.Create_GPR(TrainIn_scale,TrainOut_scale,
                                    Parameters.Kernel,prev_state=ModelFile)
    return model, likelihood, OutputScaler

def Compare(VL, DADict):
    Parameters = DADict['Parameters']
    DataFile_path = "{}/{}".format(VL.PROJECT_DIR,Parameters.DataFile)

    TestIn, TestOut = ML.GetMLdata(DataFile_path,'Test','Input','Output')
    TestIn_scale = ML.DataScale(TestIn,*InputScaler)
    TestIn_scale = torch.from_numpy(TestIn_scale)

    metric = 'RMSE'

    ResDict = {}
    for mldir in Parameters.Dirs:
        path = "{}/ML/{}".format(VL.PROJECT_DIR,mldir)
        resdirs = ML.GetResPaths(path)
        for resdir in resdirs:
            if mldir.lower().startswith(('masa','qbc_var')):
                nb = 3 if getattr(Parameters,'All',False) else 1
                for i,k in enumerate(['RBF','Matern_1.5','Matern_2.5'][:nb]):

                    nm = "{}_{}".format(mldir,k)
                    if nm not in ResDict:ResDict[nm] = []
                    if os.path.basename(resdir).startswith('Model'): continue
                    _resdir = "{}/{}".format(resdir,k)
                    model,likelihood,OutputScaler = GetModel(_resdir,DataFile_path)
                    likelihood.eval(); model.eval()

                    TestOut_scale = ML.DataScale(TestOut,*OutputScaler)
                    TrainNb = len(model.train_inputs[0][0].numpy())
                    lst = [TrainNb]
                    for j,mod in enumerate(model.models):
                        TestMetrics = ML.GetMetrics(mod,TestIn_scale,TestOut_scale[:,j])
                        lst.append(TestMetrics[2])

                    ResDict[nm].append(lst)

            else:
                model,likelihood,OutputScaler = GetModel(resdir,DataFile_path)
                if mldir not in ResDict:ResDict[mldir] = []
                likelihood.eval(); model.eval()

                TestOut_scale = ML.DataScale(TestOut,*OutputScaler)
                TrainNb = len(model.train_inputs[0][0].numpy())
                lst = [TrainNb]
                for j,mod in enumerate(model.models):
                    TestMetrics = ML.GetMetrics(mod,TestIn_scale,TestOut_scale[:,j])
                    lst.append(TestMetrics[2])

                ResDict[mldir].append( lst)

    fig = plt.figure()
    for key, val in ResDict.items():
        npval = np.array(val).T
        plt.plot(npval[0,1:],npval[1:,1:].sum(axis=0),marker='x',label=key)
    plt.ylabel(metric)
    plt.legend()
    plt.xlabel('Nb Training Points')
    plt.show()

    # fig, axs = plt.subplots(2)
    # for i, name in enumerate(OutTag):
    #     for key, val in ResDict.items():
    #         npval = np.array(val).T
    #         axs[i].plot(npval[0,1:],npval[i+1,1:],marker='x',label=key)
    #     axs[i].set_title(name.capitalize())
    #     axs[i].set_ylabel(metric)
    #     axs[i].legend()
    # axs[-1].set_xlabel('Nb Training Points')
    # plt.show()
