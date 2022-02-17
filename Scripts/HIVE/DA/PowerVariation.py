import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from types import SimpleNamespace as Namespace
import torch
import gpytorch
import time

from VLFunctions import ReadData, ReadParameters
import ML
from Optimise import FuncOpt

dtype = 'float64' # float64 is more accurate for optimisation purposes
torch_dtype = getattr(torch,dtype)
InTag, OutTag = ['x','y','z','rotation'],['power','uniformity']

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

    CompileData = getattr(Parameters,'CompileData',None)
    if CompileData:
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
                                        Parameters.Kernel,min_noise=min_noise,prev_state=ModelFile)

        ML.GPR_Train_Multi(model, Parameters.Epochs)

        torch.save(model.state_dict(), ModelFile)

        for mod in model.models:
            print('Lengthscale:',mod.covar_module.base_kernel.lengthscale.detach().numpy()[0])
            print('Outputscale', mod.covar_module.outputscale.detach().numpy())
            print('Noise',mod.likelihood.noise.detach().numpy()[0])
            print()

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

    # =========================================================================
    # Get error metrics for model

    TestMetrics = ML.GetMetrics(model,TestIn_scale,TestOut_scale.detach().numpy())
    TrainMetrics = ML.GetMetrics(model,TrainIn_scale,TrainOut_scale.detach().numpy())

    outmess = "    {}\nMSE: {}\nMAE: {}\nRMSE: {}\nR^2: {}\n"
    print(outmess.format(OutTag, TrainMetrics['MSE'],TrainMetrics['MAE'],
                         TrainMetrics['RMSE'],TrainMetrics['Rsq']))
    print(outmess.format(OutTag, TestMetrics['MSE'],TestMetrics['MAE'],
                         TestMetrics['RMSE'],TestMetrics['Rsq']))

    DADict['Data']['TestMetrics'] = TestMetrics

    bounds = [[0,1]]*NbInput
    Adaptive = getattr(Parameters,'Adaptive',{})
    if Adaptive:
        Method = Adaptive['Method']
        NbNext = Adaptive['Nb']
        NbCand = Adaptive['NbCandidates']
        maximise = Adaptive.get('Maximise',None)

        Seed = Adaptive.get('Seed',None)
        if Seed!=None: np.random.seed(Seed)
        Candidates = np.random.uniform(0,1,size=(NbCand,NbInput))
        # Candidates = ML.LHS_Samples([[0.001,0.999]]*NbInput,NbCand,Seed,100)
        # Candidates = np.array(Candidates)
        OrigCandidates = np.copy(Candidates)

        sort=True
        BestPoints = []
        tm = 0
        for i in range(NbNext):
            st = time.time()

            if not maximise:
                score, srtCandidates = ML.Adaptive(Candidates,model,Method,
                                                scoring='sum',sort=sort)
                BestPoint = srtCandidates[0:1]
                Candidates = srtCandidates #srtCandidates[1:]

            elif maximise.lower()=='slsqp':
                constr_rad = Adaptive.get('Constraint',0)
                if constr_rad:
                    constraint = ML.ConstrainRad(OrigCandidates,constr_rad)
                else: constraint = []

                score,srtCandidates = ML.AdaptSLSQP(Candidates,model,Method,[[0,1]]*NbInput,
                                                    constraints=constraint,
                                                    scoring='sum',sort=sort)
                BestPoint = srtCandidates[0:1]
                Candidates = srtCandidates #srtCandidates[1:]

            elif maximise.lower()=='ga':
                score,BestPoint = ML.AdaptGA(model,Method,bounds)
                BestPoint = np.atleast_2d(BestPoint)

            end1 = time.time()-st
            tm+=end1

            Show=0
            if Show:
                for i,j in zip(score[:Show],srtCandidates):
                    print(j,i)
                print()

            #d = np.linalg.norm(TrainIn_scale.detach().numpy() - BestPoint,axis=1)

            # Add best point to list
            BestPoints.append(BestPoint.flatten())

            # Update model with best point & mean value for better predictions
            BestPoint_pth = torch.from_numpy(BestPoint)
            with torch.no_grad():
                output = model(*[BestPoint_pth]*NbOutput)
            for j,mod in enumerate(model.models):
                _mod = mod.get_fantasy_model(BestPoint_pth,output[j].mean)
                model.models[j] = _mod

        print('Total time:',tm)
        print(np.around(BestPoints,3))
        BestPoints = ML.DataRescale(np.array(BestPoints),*InputScaler)

        DADict['Data']['BestPoints'] = BestPoints

    np.random.seed(123)
    NbInit = 50
    init_guess = np.random.uniform(0,1,size=(NbInit,4))

    # ==========================================================================
    # Get min and max values for each
    RangeDict = {}
    for i, name in enumerate(OutTag):
        for tp,order in zip(['Max','Min'],['decreasing','increasing']):
            Optima = FuncOpt(GPR_Opt, init_guess, find=tp, tol=0.01,
                             order=order,
                             bounds=bounds, jac=True, args=[model.models[i]])
            MaxPower_cd,MaxPower_val = Optima
            Opt_cd = ML.DataRescale(Optima[0],*InputScaler)
            Opt_val = ML.DataRescale(Optima[1],*OutputScaler[:,i])
            mess = "{} {}:\n{}, {}\n".format(tp,name.capitalize(),Opt_cd[0],Opt_val[0])
            print(mess)
            RangeDict["{}_{}".format(tp,name)] = Opt_val[0]



    # ==========================================================================
    # Get minimum variation for different powers

    '''
    space = 50
    rdlow = int(np.ceil(RangeDict['Min_Power'] / space)) * space
    rdhigh = int(np.ceil(RangeDict['Max_Power'] / space)) * space
    vals = []
    for i in range(rdlow,rdhigh,space):
        iscale = ML.DataScale(i,*OutputScaler[:,0])
        con = {'type': 'ineq', 'fun': constraint,
               'jac':dconstraint, 'args':(model.models[0], iscale)}
        Optima = FuncOpt(GPR_Opt, init_guess, find='min', tol=0.01,
                         order='increasing', constraints=con,
                         bounds=bnds, jac=True, args=[model.models[1]])

        # print(i)
        # with torch.no_grad():
        #     x = torch.tensor(Optima[0])
        #     out = model(*[x]*NbOutput)
        #     Power = ML.DataRescale(out[0].mean.numpy(),*OutputScaler[:,0])
        #     Variation = ML.DataRescale(out[1].mean.numpy(),*OutputScaler[:,1])
        #
        #     for P,V in zip(Power,Variation):
        #         print(P,V)
        # print()
        Opt_cd = ML.DataRescale(Optima[0],*InputScaler)
        Opt_val = ML.DataRescale(Optima[1],*OutputScaler[:,1])
        mess = 'Minimised variaition for power above {} W:\n{}, {}\n'.format(i,Opt_cd[0],Opt_val[0])
        print(mess)
        vals.append([i,Opt_val[0]])

    vals = np.array(vals)
    plt.figure()
    plt.xlabel('Power')
    plt.ylabel('Variation')
    plt.plot(vals[:,0],vals[:,1])
    plt.show()

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

def GPR_Opt(X,model):
    X = torch.tensor(X,dtype=torch_dtype)
    dmean, mean = model.Gradient_mean(X)
    return mean.detach().numpy(), dmean.detach().numpy()

def constraint(X, model, DesPower):
    '''
    Constraint which must be met during the optimisation of func. This specifies
    that the power must be greater than or equal to 'DesPower'
    '''
    X = torch.tensor(np.atleast_2d(X),dtype=torch_dtype)
    # Function value
    Pred = model(X).mean.detach().numpy()
    constr = Pred - DesPower
    return constr

def dconstraint(X, model, DesPower):
    '''
    Constraint which must be met during the optimisation of func. This specifies
    that the power must be greater than or equal to 'DesPower'
    '''
    X = torch.tensor(np.atleast_2d(X),dtype=torch_dtype)
    # Gradient
    Grad = model.Gradient(X)

    return Grad


def Compare(VL, DADict):
    Parameters = DADict['Parameters']
    DataFile_path = "{}/{}".format(VL.PROJECT_DIR,Parameters.DataFile)


    DispX = DispY = [-0.01,0.01]
    DispZ,Rotation = [0.01,0.03],[-15,15]
    bounds = np.transpose([DispX,DispY,DispZ,Rotation]) # could import this for consistency
    InputScaler = np.array([bounds[0],bounds[1] - bounds[0]])

    ResDict = {}
    for mldir in Parameters.Dirs:
        if mldir not in ResDict:ResDict[mldir] = []
        path = "{}/ML/{}".format(VL.PROJECT_DIR,mldir)
        resdirs = ML.GetResPaths(path)
        for resdir in resdirs:
            paramfile = "{}/Parameters.py".format(resdir)
            Parameters = ReadParameters(paramfile)

            TrainIn, TrainOut = ML.GetMLdata(DataFile_path, Parameters.TrainData,
                                            'Input', 'Output', Parameters.TrainNb)
            TestIn, TestOut = ML.GetMLdata(DataFile_path, Parameters.TestData,
                                           'Input', 'Output', Parameters.TestNb)

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

            NbInput,NbOutput = TrainIn.shape[1],TrainOut.shape[1]
            TrainNb = len(TrainIn)
            ModelFile = "{}/Model.pth".format(resdir)
            likelihood, model = ML.GPRModel_Multi(TrainIn_scale,TrainOut_scale,
                                            Parameters.Kernel,prev_state=ModelFile)
            likelihood.eval()
            model.eval()
            reslst = [TrainNb]
            with torch.no_grad():
                TrainPred = model(*[TrainIn_scale]*NbOutput)
                TestPred = model(*[TestIn_scale]*NbOutput)

                for i in range(NbOutput):
                    Train_mean = TrainPred[i].mean.numpy()
                    Test_mean = TestPred[i].mean.numpy()

                    TrainMSE = ML.MSE(Train_mean,TrainOut_scale[:,i].numpy())
                    TestMSE = ML.MSE(Test_mean,TestOut_scale[:,i].numpy())
                    reslst.append(TestMSE)

            ResDict[mldir].append(reslst)

    plt.figure()
    for key, val in ResDict.items():
        npval = np.array(val).T
        plt.plot(npval[0],npval[1],marker='x',label=key[:-4])
    plt.xlabel('Nb Training Points')
    plt.ylabel('Power MSE')
    plt.legend()
    plt.show()

    plt.figure()
    for key, val in ResDict.items():
        npval = np.array(val).T
        plt.plot(npval[0],npval[2],marker='x',label=key[:-4])
    plt.xlabel('Nb Training Points')
    plt.ylabel('Variation MSE')
    plt.legend()
    plt.show()
