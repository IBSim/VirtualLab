import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace as Namespace
import torch
import gpytorch
from PIL import Image

from VLFunctions import ReadData, ReadParameters
from Scripts.Common.ML import ML, Adaptive
from Scripts.Common.ML.slsqp_multi import slsqp_multi

dtype = 'float64' # float64 is more accurate for optimisation purposes
torch_dtype = getattr(torch,dtype)
torch.set_default_dtype(torch_dtype)

OutputLabels = ['Power','Variation']

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

    CmpData = Parameters.CompileData
    if type(CmpData)==str:CmpData = [CmpData]

    ResDirs = ["{}/{}".format(VL.PROJECT_DIR,resname) for resname in CmpData]

    InData, OutData = ML.CompileData(ResDirs,MLMapping)

    ML.WriteMLdata(DataFile_path, CmpData, Parameters.InputName,InData)
    ML.WriteMLdata(DataFile_path, CmpData, Parameters.OutputName, OutData)

# ==============================================================================
# default function
def Single(VL, DADict):
    Parameters = DADict["Parameters"]

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
    TrainNb,TestNb = getattr(Parameters,'TrainNb',-1),getattr(Parameters,'TestNb',-1)
    TrainIn, TrainOut = ML.GetMLdata(DataFile_path, Parameters.TrainData,
                                    InputName, OutputName, TrainNb)
    TestIn, TestOut = ML.GetMLdata(DataFile_path, Parameters.TestData,
                                   InputName, OutputName, TestNb)

    NbInput,NbOutput = TrainIn.shape[1],TrainOut.shape[1]

    PS_bounds = np.array(Parameters.ParameterSpace).T
    InputScaler = ML.ScaleValues(PS_bounds)

    # Scale input to [0,1] (based on parameter space)
    TrainIn_scale = ML.DataScale(TrainIn,*InputScaler)
    TestIn_scale = ML.DataScale(TestIn,*InputScaler)
    # Scale output to [0,1] (based on training data)
    OutputScaler = ML.ScaleValues(TrainOut)
    TrainOut_scale = ML.DataScale(TrainOut,*OutputScaler)
    TestOut_scale = ML.DataScale(TestOut,*OutputScaler)

    # Convert to tensors
    TrainIn_scale = torch.from_numpy(TrainIn_scale)
    TrainOut_scale = torch.from_numpy(TrainOut_scale)

    TestIn_scale = torch.from_numpy(TestIn_scale)
    TestOut_scale = torch.from_numpy(TestOut_scale)

    if hasattr(Parameters,'FeatureNames'):FeatureNames = Parameters.FeatureNames
    else: FeatureNames = ["Feature_{}".format(i) for i in range(NbInput)]

    # ==========================================================================
    # Model summary
    ML.ModelSummary(NbInput,NbOutput,TrainNb,TestNb,FeatureNames,OutputLabels)

    # ==========================================================================
    # Train a new model or load an old one
    ModelFile = "{}/Model.pth".format(DADict['CALC_DIR']) # Saved model location
    if Parameters.Train:
        # get model & likelihoods
        min_noise = getattr(Parameters,'MinNoise',None)
        likelihood, model = ML.GPRModel_Multi(TrainIn_scale, TrainOut_scale,
                                        Parameters.Kernel,min_noise=min_noise)
        # Train model
        ML.GPR_Train_Multi(model, Parameters.Epochs)
        # Save model
        torch.save(model.state_dict(), ModelFile)
        # Print model parameters for each output
        for mod in model.models:
            print('Lengthscale:',mod.covar_module.base_kernel.lengthscale.detach().numpy()[0])
            print('Outputscale', mod.covar_module.outputscale.detach().numpy())
            print('Noise',mod.likelihood.noise.detach().numpy()[0])
            print()
    else:
        # Load previously trained model
        likelihood, model = ML.GPRModel_Multi(TrainIn_scale,TrainOut_scale,
                                        Parameters.Kernel,prev_state=ModelFile)
    model.eval();likelihood.eval()

    # =========================================================================
    # Get error metrics for model
<<<<<<< HEAD
    TestMetrics, TrainMetrics = [], []
    for i, mod in enumerate(model.models):
        train = ML.GetMetrics(mod, TrainIn_scale, TrainOut_scale.detach().numpy()[:,i])
        test = ML.GetMetrics(mod, TestIn_scale, TestOut_scale.detach().numpy()[:,i])
        TrainMetrics.append(train); TestMetrics.append(test)
    TestMetrics, TrainMetrics = np.array(TestMetrics).T, np.array(TrainMetrics).T

    for tp,data in zip(['Train','Test'],[TrainMetrics,TestMetrics]):
        outstr = "{} Data\n    {}   {}\n".format(tp,*OutTag)
        for i,metric in enumerate(['MSE','MAE','RMSE','R^2']):
            outstr+="{}: {}\n".format(metric,data[i])
        print(outstr)
=======
    with torch.no_grad():
        train_pred_scale = model(*[TrainIn_scale]*NbOutput)
        test_pred_scale = model(*[TestIn_scale]*NbOutput)

    train_pred_scale = np.transpose([p.mean.numpy() for p in train_pred_scale])
    test_pred_scale = np.transpose([p.mean.numpy() for p in test_pred_scale])
    train_pred = ML.DataRescale(train_pred_scale,*OutputScaler)
    test_pred = ML.DataRescale(test_pred_scale,*OutputScaler)

    df_train = ML.GetMetrics2(train_pred,TrainOut)
    df_test = ML.GetMetrics2(test_pred,TestOut)
    print('\nTrain metrics')
    print(df_train)
    print('\nTest metrics')
    print(df_test,'\n')
>>>>>>> Update to Power variation work

    # ==========================================================================
    # Get next points to collect data
    bounds = [[0,1]]*NbInput

    AdaptDict = getattr(Parameters,'Adaptive',{})
    if AdaptDict:
        BestPoints = Adaptive.Adaptive(model, AdaptDict, bounds,Show=5)
        print(np.around(BestPoints,3))
        BestPoints = ML.DataRescale(np.array(BestPoints),*InputScaler)
        DADict['Data']['BestPoints'] = BestPoints


    # ==========================================================================
    # Get min and max values for each
    print('Extrema')
    np.set_printoptions(precision=3)
    RangeDict = {}
    for i, mod in enumerate(model.models):
        val,cd = ML.GetExtrema(mod, 50, bounds)
        cd = ML.DataRescale(cd,*InputScaler)
        val = ML.DataRescale(val,*OutputScaler[:,i])
        RangeDict["Min_{}".format(OutputLabels[i])] = val[0]
        RangeDict["Max_{}".format(OutputLabels[i])] = val[1]
        print(OutputLabels[i])
        print("Minima: Value:{:.2f}, Input: {}".format(val[0],cd[0]))
        print("Maxima: Value:{:.2f}, Input: {}\n".format(val[1],cd[1]))


    # ==========================================================================
    # Get minimum variation for different powers & plot
<<<<<<< HEAD

    space = 100
    rdlow = int(np.ceil(RangeDict['Min_Power'] / space)) * space
    rdhigh = int(np.ceil(RangeDict['Max_Power'] / space)) * space
    con = {'type': 'ineq', 'fun': MinPower, 'jac':dMinPower}
    P,V = [],[]
    for i in range(rdlow,rdhigh,space):
        iscale = ML.DataScale(i,*OutputScaler[:,0])
        con['args'] = (model.models[0], iscale)
        Opt_cd, Opt_val = ML.GetOptima(model.models[1], 50, bounds,
                                       find='min', order='increasing',
                                       constraints=con)

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
=======
    MinVar_Constraint = getattr(Parameters,'MinVar_Constraint',{})
    if MinVar_Constraint:
        space = MinVar_Constraint['Space']
        rdlow = int(np.ceil(RangeDict['Min_Power'] / space)) * space
        rdhigh = int(np.ceil(RangeDict['Max_Power'] / space)) * space
        # add min values to P & V array
        P,V = [RangeDict['Min_Power']],[RangeDict['Min_Variation']]
        for i in range(rdlow,rdhigh,space):
            iscale = ML.DataScale(i,*OutputScaler[:,0])
            con = ML.LowerBound(model.models[0], iscale)
            Opt_cd, Opt_val = ML.GetOptima(model.models[1], 100, bounds,
                                           find='min', constraints=con, maxiter=30)

            Opt_cd = ML.DataRescale(Opt_cd,*InputScaler)
            Opt_val = ML.DataRescale(Opt_val,*OutputScaler[:,1])

            mess = 'Minimised variaition for power above {} W:\n{}, {:.2f}\n'.format(i,Opt_cd[0],Opt_val[0])
            print(mess)
            P.append(i);V.append(Opt_val[0])
        # Add Max Power point
        P.append(RangeDict['Max_Power']); V.append(RangeDict['Max_Variation'])
        if MinVar_Constraint['Plot']:
            plt.figure()
            plt.xlabel('Power')
            plt.ylabel('Variation')
            plt.plot(P,V)
            plt.savefig("{}/Variation_constrained.png".format(DADict['CALC_DIR']))
            plt.close()

>>>>>>> Update to Power variation work

    # ==========================================================================
    # See impact of varying inputs
    Plot1D = getattr(Parameters,'Plot1D',{})
    if Plot1D:
        base = Plot1D['Base']
        ncol = Plot1D.get('NbCol',1)

        for j, mod in enumerate(model.models):
            mean,stdev = ML.InputQuery(mod,NbInput,base=0.5)
            nrow = int(np.ceil(NbInput/ncol))
            fig,ax = plt.subplots(nrow,ncol,figsize=(15,15))
            axes = ax.flatten()

            base_in = [base]*NbInput
            with torch.no_grad():
                base_val = mod(torch.tensor([base_in])).mean.numpy()

            base_in = ML.DataRescale(np.array(base_in),*InputScaler)
            base_val = ML.DataRescale(base_val,*OutputScaler[:,j])

            for i, (val,std) in enumerate(zip(mean,stdev)):
                val = ML.DataRescale(val,*OutputScaler[:,j])
                std = ML.DataRescale(std,0,OutputScaler[1,j])
                axes[i].title.set_text(FeatureNames[i])
                _split = np.linspace(InputScaler[0,i],InputScaler[:,i].sum(),len(val))

                axes[i].plot(_split,val)
                axes[i].fill_between(_split, val-2*std, val+2*std, alpha=0.5)
                axes[i].scatter(base_in[i],base_val)

            fig.text(0.5, 0.04, 'Parameter range', ha='center')
            fig.text(0.04, 0.5, OutputLabels[j], va='center', rotation='vertical')

            plt.show()

    # ==========================================================================
    # See slices of parameter space
    Plot4D = getattr(Parameters,'Plot4D',{})
    if Plot4D:
        Query = Plot4D['Query']
        disc,fixed = [],[]
        for i, q in enumerate(Query):
            if type(q) in (int,float):
                a = np.linspace(0+0.5*1/int(q),1-0.5*1/int(q),int(q))
            else:
                a = ML.DataScale(np.array(q),*InputScaler[:,i])
                fixed.append(i)
            disc.append(a)

        grid = np.meshgrid(*disc, indexing='ij')
        grid = np.moveaxis(grid,0,-1) #grid point is now the last axis
        grid_unroll = grid.reshape((int(grid.size/NbInput),NbInput))
        grid_unroll = torch.tensor(grid_unroll, dtype=torch_dtype)

        tiff_dir = "{}/Tiff".format(DADict['CALC_DIR'])
        os.makedirs(tiff_dir,exist_ok=True)

        sl = [slice(None)]*NbInput
        for ix in fixed: sl[ix]=0

        for j, mod in enumerate(model.models):
            with torch.no_grad():
                out_grid = mod(grid_unroll).mean.numpy()
            out_grid = ML.DataRescale(out_grid,*OutputScaler[:,j])
            out_grid = out_grid.reshape(grid.shape[:-1])
            out_grid = out_grid[tuple(sl)]
            for k in range(out_grid.shape[0]):
                data_tif = Image.fromarray(out_grid[k])
                data_tif.save("{}/{}_{}.tif".format(tiff_dir,OutputLabels[j],k))

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
        likelihood, model = ML.GPRModel_Multi(TrainIn_scale,TrainOut_scale,
                                        ModParameters.Kernel,prev_state=ModelFile)

        likelihood.eval();model.eval()
        Likelihoods.append(likelihood)
        Models.append(model)

    bounds = [[0,1]]*len(TrainIn.shape[1])
    AdaptDict = getattr(Parameters,'Adaptive',{})
    if AdaptDict:
        st = time.time()
        BestPoints = Adaptive.Adaptive(Models, AdaptDict, bounds,Show=3)
        end = time.time() - st
        print('Time',end)
        print(np.around(BestPoints,3))
        BestPoints = ML.DataRescale(np.array(BestPoints),*InputScaler)
        DADict['Data']['BestPoints'] = BestPoints

# ==============================================================================
<<<<<<< HEAD
'''
Constraint . This specifies
that the power must be greater than or equal to 'DesPower'
'''
def MinPower(X, model, DesPower):
    X = torch.tensor(np.atleast_2d(X),dtype=torch_dtype)
    # Function value
    Pred = model(X).mean.detach().numpy()
    constr = Pred - DesPower
    return constr

def dMinPower(X, model, DesPower):
    X = torch.tensor(np.atleast_2d(X),dtype=torch_dtype)
    # Gradient
    Grad = model.Gradient(X)
    return Grad

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

=======
>>>>>>> Update to Power variation work
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
    likelihood, model = ML.GPRModel_Multi(TrainIn_scale,TrainOut_scale,
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
    # for i, name in enumerate(OutputLabels):
    #     for key, val in ResDict.items():
    #         npval = np.array(val).T
    #         axs[i].plot(npval[0,1:],npval[i+1,1:],marker='x',label=key)
    #     axs[i].set_title(name.capitalize())
    #     axs[i].set_ylabel(metric)
    #     axs[i].legend()
    # axs[-1].set_xlabel('Nb Training Points')
    # plt.show()
