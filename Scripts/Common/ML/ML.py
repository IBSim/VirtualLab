import os
import sys
import time
import pandas as pd

import numpy as np
import torch
import gpytorch
import h5py
from natsort import natsorted

from Scripts.Common.Optimisation import slsqp_multi

class ExactGPmodel(gpytorch.models.ExactGP):
    '''
    Gaussian process regression model.
    '''
    def __init__(self, train_x, train_y, likelihood, kernel,options={},ard=True):
        super(ExactGPmodel, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()

        ard_num_dims = train_x.shape[1] if ard else None
        if kernel.lower() in ('rbf'):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims))
        if kernel.lower().startswith('matern'):
            split = kernel.split('_')
            nu = float(split[1]) if len(split)==2 else 2.5
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=nu,ard_num_dims=ard_num_dims))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def Gradient(self, x):
        x.requires_grad=True
        with gpytorch.settings.fast_pred_var():
            # pred = self.likelihood(self(x))
            pred = self(x)
            grads = torch.autograd.grad(pred.mean.sum(), x)[0]
            return grads

    def Gradient_mean(self, x):
        x.requires_grad=True
        # pred = self.likelihood(self(x))
        mean = self(x).mean
        dmean = torch.autograd.grad(mean.sum(), x)[0]
        return dmean, mean

    def Gradient_variance(self, x):
        x.requires_grad=True
        with gpytorch.settings.fast_pred_var():
            # pred = self.likelihood(self(x))
            var = self(x).variance
            dvar = torch.autograd.grad(var.sum(), x)[0]
        return dvar, var

def Create_GPR(TrainIn,TrainOut,Kernel,prev_state=None,min_noise=None):
    if TrainOut.ndim==1:
        # single output
        likelihood, model = _Create_GPR(TrainIn, TrainOut, Kernel, min_noise=min_noise)
    else:
        # multiple output
        NbModel = TrainOut.shape[1]
        # Change kernel and min_noise to a list
        if type(Kernel) not in (list,tuple): Kernel = [Kernel]*NbModel
        if type(min_noise) not in (list,tuple): min_noise = [min_noise]*NbModel
        models,likelihoods = [], []
        for i in range(NbModel):
            likelihood, model = _Create_GPR(TrainIn, TrainOut[:,i], Kernel[i],
                                            min_noise = min_noise[i])
            models.append(model)
            likelihoods.append(likelihood)

        model = gpytorch.models.IndependentModelList(*models)
        likelihood = gpytorch.likelihoods.LikelihoodList(*likelihoods)

    if prev_state:
        # Check that the file exists
        if os.path.isfile(prev_state):
            state_dict = torch.load(prev_state)
            model.load_state_dict(state_dict)
        else:
            print('Warning\nPrevious state file doesnt exist\nNo previous model is loaded\n')

    return likelihood, model

def _Create_GPR(TrainIn,TrainOut,Kernel,prev_state=None,min_noise=None):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPmodel(TrainIn, TrainOut, likelihood, Kernel)
    if not prev_state:
        if min_noise != None:
            likelihood.noise_covar.register_constraint('raw_noise',gpytorch.constraints.GreaterThan(min_noise))

        # Start noise at lower level to avoid bad optima, but ensure it isn't zero
        noise_lower = likelihood.noise_covar.raw_noise_constraint.lower_bound
        noise_init = max(5*noise_lower,1e-8)
        hypers = {'likelihood.noise_covar.noise': noise_init}
        model.initialize(**hypers)
    else:
        state_dict = torch.load(prev_state)
        model.load_state_dict(state_dict)

    return likelihood, model

# ==============================================================================
# Train the GPR model
def GPR_Train(model, Epochs=5000, lr=0.01, Print=50, ConvAvg=10, tol=1e-4,
              Verbose=False, SumOutput=False):

    likelihood = model.likelihood
    model.train()
    likelihood.train()

    MultiOutput = True if hasattr(model,'models') else False
    # Create the necessary loss functions and optimisers
    if MultiOutput and not SumOutput:
        # Each output of the model is trained seperately
        TrackLoss, Completed = 0, []
        _mll = gpytorch.mlls.ExactMarginalLogLikelihood
        LossFn,Losses, optimizer = [],[],[]
        for mod in model.models:
            LossFn.append(_mll(mod.likelihood,mod))
            optimizer.append(torch.optim.Adam(mod.parameters(), lr=lr))
            Losses.append([])
    elif MultiOutput:
        # Each output is trained together
        Losses = [[]]
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        LossFn = gpytorch.mlls.SumMarginalLogLikelihood(likelihood, model)
    else:
        # Single output model
        Losses = [[]]
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        LossFn = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood,model)

    # Start looping over training data
    for i in range(Epochs):
        if MultiOutput and not SumOutput:
            TotalLoss = TrackLoss
            for j,mod in enumerate(model.models):
                if j in Completed: continue

                _Losses = Losses[j]
                _convergence = _Step(mod,optimizer[j],LossFn[j],_Losses,ConvAvg=ConvAvg,tol=tol)
                TotalLoss+=_Losses[-1]

                if _convergence != None:
                    Completed.append(j)
                    TrackLoss+=_Losses[-1]
                    print("Output {}: {}".format(j,_convergence))

            TotalLoss = TotalLoss/(j+1)
            if len(Completed)==j+1:
                break
        else:
            _Losses = Losses[0]
            convergence = _Step(model,optimizer,LossFn,_Losses,ConvAvg=ConvAvg,tol=tol)
            TotalLoss = _Losses[-1]
            if convergence != None:
                print(convergence)
                break

        if i==0 or (i+1) % Print == 0:
            print("Iteration: {}, Loss: {}".format(i+1,TotalLoss))
            if Verbose:
                PrintParameters(model)

    # Print out final information about training & model parameters
    print('\n################################\n')
    print("Iterations: {}\nLoss: {}".format(i+1,TotalLoss))
    print("\nModel parameters:")
    PrintParameters(model)
    print('################################\n')

    return Losses

def _Step(model, optimizer, mll, loss_lst, ConvAvg=10, tol=1e-4):
    optimizer.zero_grad() # set all gradients to zero
    # Calculate loss & add to list
    output = model(*model.train_inputs)
    loss = -mll(output, model.train_targets)
    loss_lst.append(loss.item())
    # Check convergence using the loss list. If convergence, return
    convergence = CheckConvergence(loss_lst,ConvAvg=ConvAvg,tol=tol)
    if convergence != None:
        return convergence

    # Calculate gradients & update model parameters using the optimizer
    loss.backward()
    optimizer.step()

def CheckConvergence(Loss, ConvAvg=10, tol=1e-4):
    ''' Checks the list of loss values to decide whether or not convergence has been reached'''
    if len(Loss) >= 2*ConvAvg:
        mean_new = np.mean(Loss[-ConvAvg:])
        mean_old = np.mean(Loss[-2*ConvAvg:-ConvAvg])
        if mean_new > mean_old:
            return "Convergence reached. Loss increasing"
        elif np.abs(mean_new-mean_old)<tol:
            return "Convergence reached. Loss change smaller than tolerance"

def PrintParameters(model):
    Modstr = "Length scales: {}\nOutput scale: {}\nNoise: {}\n\n"
    if hasattr(model,'models'):
        Rstr = ""
        for i,mod in enumerate(model.models):
            LS = mod.covar_module.base_kernel.lengthscale.detach().numpy()[0]
            OS = mod.covar_module.outputscale.detach().numpy()
            N = mod.likelihood.noise.detach().numpy()[0]
            Rstr += ("Output {}\n"+Modstr).format(i,LS,OS,N)
    else:
            LS = model.covar_module.base_kernel.lengthscale.detach().numpy()[0]
            OS = model.covar_module.outputscale.detach().numpy()
            N = model.likelihood.noise.detach().numpy()[0]
            Rstr = Modstr.format(LS,OS,N)

    print(Rstr,end='')

# ==============================================================================
# Data scaling and rescaling functions
def ScaleValues(data,scaling='unit'):
    ''' '''
    if scaling.lower()=='unit':
        datamin,datamax = data.min(axis=0),data.max(axis=0)
        scaler = np.array([datamin,datamax-datamin])
    return scaler

def DataScale(data,const,scale):
    '''
    This function scales n-dim data to a specific range.
    data: N-darray or scalar
    const: N-darray or scalar
    scale: N-darray or scalar
    Examples:
     - Normalising data:
        const=mean, scale=stddev
     - [0,1] range:
        const=min, scale=max-min
    '''
    return (data - const)/scale

def DataRescale(data,const,scale):
    '''
    This function scales data back to original range.
    data: N-darray or scalar
    const: N-darray or scalar
    scale: N-darray or scalar
    '''
    return data*scale + const

# ==============================================================================
# Metrics used to asses model performance
def MSE(Predicted,Target):
    # this is not normnalised
    sqdiff = (Predicted - Target)**2
    return np.mean(sqdiff,axis=0)

def MAE(Predicted,Target):
    return np.abs(Predicted - Target).mean(axis=0)/(Target.max(axis=0) - Target.min(axis=0))

def RMSE(Predicted,Target):
    return ((Predicted - Target)**2).mean(axis=0)**0.5/(Target.max(axis=0) - Target.min(axis=0))

def Rsq(Predicted,Target):
    mean_pred = Predicted.mean(axis=0)
    divisor = ((Predicted - mean_pred)**2).sum(axis=0)
    MSE_val = ((Predicted - Target)**2).sum(axis=0)
    return 1-(MSE_val/divisor)

def _GetMetrics(model,x,target):
    with torch.no_grad():
        pred = model(x).mean.numpy()
    mse = MSE(pred,target)
    mae = MAE(pred,target)
    rmse = RMSE(pred,target)
    rsq = Rsq(pred,target)
    return mse,mae,rmse,rsq

def GetMetrics(model,x,target):
    if hasattr(model,'models'):
        mse,mae,rmse,rsq=[],[],[],[]
        for i,mod in enumerate(model.models):
            _mse,_mae,_rmse,_rsq = _GetMetrics(mod,x,target[:,i])
            mse.append(_mse);mae.append(_mae);
            rmse.append(_rmse);rsq.append(_rsq);
    else:
        mse,mae,rmse,rsq = _GetMetrics(model,x,target)

    df=pd.DataFrame({"MSE":mse,"MAE":mae,"RMSE":rmse,"R^2":rsq},
                    index=["Output_{}".format(i) for i in range(len(mse))])
    pd.options.display.float_format = '{:.3e}'.format
    return df

def GetMetrics2(pred,target):
    mse = MSE(pred,target)
    mae = MAE(pred,target)
    rmse = RMSE(pred,target)
    rsq = Rsq(pred,target)

    df=pd.DataFrame({"MSE":mse,"MAE":mae,"RMSE":rmse,"R^2":rsq},
                    index=["Output_{}".format(i) for i in range(len(mse))])
    pd.options.display.float_format = '{:.3e}'.format
    return df


# ==============================================================================
# Functions used for reading & writing data
def GetResPaths(ResDir,DirOnly=True,Skip=['_']):
    ''' This returns a naturally sorted list of the directories in ResDir'''
    ResPaths = []
    for _dir in natsorted(os.listdir(ResDir)):
        if _dir.startswith(tuple(Skip)): continue
        path = "{}/{}".format(ResDir,_dir)
        if DirOnly and os.path.isdir(path):
            ResPaths.append(path)

    return ResPaths

def Openhdf(File,style,timer=5):
    ''' Repeatedly attemps to open hdf file if it is held by another process for
    the time allocated by timer '''
    st = time.time()
    while True:
        try:
            Database = h5py.File(File,style)
            return Database
        except OSError:
            if time.time() - st > timer:
                sys.exit('Timeout on opening hdf file')

def Writehdf(File, array, data_path):
    Database = Openhdf(File,'a')
    if data_path in Database:
        del Database[data_path]
    Database.create_dataset(data_path,data=array)
    Database.close()

def Readhdf(File, data_paths):
    Database = Openhdf(File,'r')

    if type(data_paths)==str: data_paths = [data_paths]
    data = []
    for data_path in data_paths:
        _data = Database[data_path][:]
        data.append(_data)
    Database.close()
    return data

def GetMLdata(DataFile_path,DataNames,InputName,OutputName,Nb=-1):
    if type(DataNames)==str:DataNames = [DataNames]
    N = len(DataNames)

    data_input, data_output = [],[]
    for dataname in DataNames:
        data_input.append("{}/{}".format(dataname,InputName))
        data_output.append("{}/{}".format(dataname,OutputName))

    Data = Readhdf(DataFile_path,data_input+data_output)
    In,Out = Data[:N],Data[N:]

    for i in range(N):
        _Nb = Nb[i] if type(Nb)==list else Nb
        if _Nb==-1:continue

        if type(_Nb)==int:
            In[i] = In[i][:_Nb]
            Out[i] = Out[i][:_Nb]
        if type(_Nb) in (list,tuple):
            l,u = _Nb
            In[i] = In[i][l:u]
            Out[i] = Out[i][l:u]
    In,Out = np.vstack(In),np.vstack(Out)

    return In, Out

def WriteMLdata(DataFile_path,DataNames,InputName,OutputName,InList,OutList):
    for resname, _in, _out in zip(DataNames, InList, OutList):
        InPath = "{}/{}".format(resname,InputName)
        OutPath = "{}/{}".format(resname,OutputName)
        Writehdf(DataFile_path,_in,InPath)
        Writehdf(DataFile_path,_out,OutPath)

def CompileData(ResDirs,MapFnc,args=[]):
    In,Out = [],[]
    for ResDir in ResDirs:
        ResPaths = GetResPaths(ResDir)
        _In, _Out =[] ,[]
        for ResPath in ResPaths:
            _in, _out = MapFnc(ResPath,*args)
            _In.append(_in)
            _Out.append(_out)
        In.append(_In)
        Out.append(_Out)
    return In, Out

def GetInputs(Parameters,commands):
    ''' Using exec allows us to get individual values from dictionaries or lists.
    i.e. a command of 'DataList[1]' will get the value from index 1 of the lists
    'DataList'
    '''

    inputs = []
    for i,command in enumerate(commands):
        exec("inputs.append(Parameters.{})".format(command))
    return inputs

# ==============================================================================
# ML model Optima

def GetOptima(model, NbInit, bounds,find='max',tol=0.01,order='decreasing',seed=None,
             jac=True,**kwargs):
    if seed!=None: np.random.seed(seed)
    init_points = np.random.uniform(0,1,size=(NbInit,len(bounds)))

    Optima = slsqp_multi(_GPR_Opt, init_points, bounds=bounds,
                         find=find, tol=tol, order=order,
                         jac=jac, args=[model],**kwargs)
    Optima_cd, Optima_val = Optima
    return Optima_cd, Optima_val

def GetExtrema(model,NbInit,bounds,seed=None):
    # ==========================================================================
    # Get min and max values for each
    Extrema_cd, Extrema_val = [], []
    for tp in ['min','max']:
        _Extrema_cd, _Extrema_val = GetOptima(model, NbInit, bounds,
                                              find=tp, seed=seed)
        Extrema_cd.append(_Extrema_cd[0])
        Extrema_val.append(_Extrema_val[0])
    return np.array(Extrema_val), np.array(Extrema_cd)

def _GPR_Opt(X,model):
    torch.set_default_dtype(torch.float64)
    X = torch.tensor(X)
    dmean, mean = model.Gradient_mean(X)
    return mean.detach().numpy(), dmean.detach().numpy()

# ==============================================================================
# Constraint for ML model
def LowerBound(model,bound):
    constraint_dict = {'fun': _bound, 'jac':_dbound,
                       'type': 'ineq', 'args':(model,bound)}
    return constraint_dict

def UpperBound(model,bound):
    constraint_dict = {'fun': _bound, 'jac':_dbound,
                       'type': 'ineq', 'args':(model,bound,-1)}
    return constraint_dict

def FixedBound(model,bound):
    constraint_dict = {'fun': _bound, 'jac':_dbound,
                       'type': 'eq', 'args':(model,bound)}
    return constraint_dict

def _bound(X, model, bound, sign=1):
    X = torch.tensor(np.atleast_2d(X))
    # Function value
    Pred = model(X).mean.detach().numpy()
    return sign*(Pred - bound)

def _dbound(X, model, bound, sign=1):
    X = torch.tensor(np.atleast_2d(X))
    # Gradient
    Grad = model.Gradient(X)
    return sign*Grad
