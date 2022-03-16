import os
import h5py
import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
import scipy.stats as stats

from VLFunctions import ReadData, ReadParameters
from Scripts.Common.ML import ML
from Scripts.Common.ML.slsqp_multi import slsqp_multi

dtype = 'float64' # float64 is more accurate for optimisation purposes
torch_dtype = getattr(torch,dtype)
MaxCholeskySize = 1500

DispX = DispY = [-0.01,0.01]
DispZ = [0.01,0.02]
CoolantP, CoolantT, CoolantV = [0.4,1.6], [30,70], [5,15]
Current = [600,1000]

InTag = ['Coil X','Coil Y','Coil Z', 'Coolant Pressure',
         'Coolant Temperature', 'Coolant Velocity','Coil Current']
OutTag = ['TC_{}'.format(j) for j in range(7)]

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
    Out = DataDict['TC_Temp'].flatten()
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

def Single(VL,DADict):
    Parameters = DADict['Parameters']
    if getattr(Parameters,'CompileData',None):
        CompileData(VL,DADict)

    DataFile_path = "{}/{}".format(VL.PROJECT_DIR, Parameters.DataFile)
    InputName = getattr(Parameters,'InputName','Input')
    OutputName = getattr(Parameters,'OutputName','Output')

    # ==========================================================================
    # Get Train & test data and scale
    TrainNb, TestNb = getattr(Parameters,'TrainNb',-1),getattr(Parameters,'TestNb',-1)
    TrainIn, TrainOut = ML.GetMLdata(DataFile_path, Parameters.TrainData,
                                    InputName, OutputName, TrainNb)
    TestIn, TestOut = ML.GetMLdata(DataFile_path, Parameters.TestData,
                                   InputName, OutputName, TestNb)

    # TrainIn,TestIn = TrainIn[:,:6],TestIn[:,:6]
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

    np.random.seed(100)
    init_points = np.random.uniform(0,1,size=(100,NbInput))

    # target_input = TestIn_scale.detach().numpy()[0,:]
    # target_output = TestOut_scale.detach().numpy()[0,:]
    # init_points[:,-1] = target_input[-1]
    # _std = variability(model,target_input,target_output,init_points,fix=[-1])
    # print(_std)
    # return

    N_avg = 10
    target_inputs = TestIn_scale.detach().numpy()[:N_avg,:]
    target_outputs = TestOut_scale.detach().numpy()[:N_avg,:]

    vars = []
    for i in range(1,len(model.models)+1):
        ixs = list(range(i))
        std_all = []
        for t_input,t_output in zip(target_inputs,target_outputs):
            if False:
                _std = variability(model,t_input,t_output,init_points,ixs=ixs)
            else:
                init_points[:,-1] = t_input[-1]
                _std = variability(model,t_input,t_output,init_points, ixs=ixs, fix=[-1])
            std_all.append(_std)
        avg_std = np.array(std_all).mean(axis=0)
        print(avg_std)
        vars.append(avg_std)
    vars = np.array(vars)

    plt.figure()
    for i in range(bounds.shape[1]):
        if not vars[:,i].any(): continue
        plt.plot(list(range(1,1+len(vars))),vars[:,i],label=InTag[i])
    plt.legend()
    plt.show()

def variability(model,target_input, target_output, init_points, ixs=None, fix=None):
    _bounds = [[0,1]]*bounds.shape[1]
    Opt_cd, Opt_val = slsqp_multi(obj, init_points, bounds=_bounds,
                                  args=[model,target_output,ixs,fix],
                                  maxiter=30, find='min', tol=0, jac=True)

    std = np.mean((Opt_cd - target_input)**2,axis=0)
    # std = np.std(Opt_cd,axis=0).flatten()

    if False:
        # print(Opt_val.min(),Opt_val.max(),'\n')
        print('##############################################')
        print('Output')
        print('Target')
        print(target_output)
        print('Model')
        print(Test_predictions[j])
        # print(((target_output - Test_predictions[j])**2).sum())
        with torch.no_grad():
            _Opt_cd = torch.from_numpy(Opt_cd)
            out = [mod(_Opt_cd).mean.numpy() for mod in model.models]
            out = np.transpose(out)
        print('Inverse solutions')
        for _out in out:
            sc = ((_out - target_output)**2)
            print(_out,sc.sum())
            print(sc)

        print('\n##############################################')
        print('Input')
        print('Target')
        print(target_input)
        print('\nInverse solutions')
        for cd,val in zip(Opt_cd,Opt_val):
            print(cd)

    if False:
        mean = np.mean(Opt_cd,axis=0).flatten()
        for trueval,mu,sigma,name in zip(target_input,mean,std,InTag):
            x = np.linspace(mu-3*sigma,mu+3*sigma,100)
            Gauss_pdf = stats.norm.pdf(x, mu, sigma)
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_title(name.capitalize())
            ax.plot(x,Gauss_pdf)
            ax.plot([trueval]*2, [0,Gauss_pdf.max()], linestyle='--')
            ax.set_xlim([0,1])
            ax.get_yaxis().set_visible(False)
            plt.show()

    return std

def obj(X, model, Target,ixs=None,fix=None):
    if ixs == None: ixs = list(range(len(model.models)))
    X = torch.tensor(np.atleast_2d(X),dtype=torch_dtype)
    Preds, Grads = [], []
    Target = Target[ixs]
    for i, mod in enumerate(model.models):
        if i not in ixs: continue
        _Grad, _Pred = mod.Gradient_mean(X)
        Preds.append(_Pred.detach().numpy())
        Grads.append(_Grad.detach().numpy())
    Preds = np.array(Preds)
    Grads = np.swapaxes(Grads,0,1)
    if fix !=None: Grads[:,:,fix] = 0
    d = np.transpose(Preds - Target[:,None])

    Score = (d**2).sum(axis=1)
    dScore = 2*(Grads*d[:,:,None]).sum(axis=1)

    return Score, dScore
