import os
import sys
import numpy as np
import torch
import gpytorch
import h5py
from natsort import natsorted
import time

from .slsqp_multi import slsqp_multi

from . import Models

# ==============================================================================
# Functions to easily create GPR
def Create_GPR(TrainIn,TrainOut,kernel='RBF',prev_state=None,min_noise=None,input_scale=None,
                output_scale=None,multitask=False):
    if TrainOut.ndim==1:
        # single output
        likelihood, model = _SingleGPR(TrainIn, TrainOut, kernel, min_noise=min_noise,
                                        input_scale=input_scale,output_scale=output_scale)

    elif multitask:
        likelihood, model = _MultitaskGPR(TrainIn,TrainOut,kernel,min_noise=min_noise,
                                        input_scale=input_scale,output_scale=output_scale)
    else:
        # multiple output
        NbModel = TrainOut.shape[1]
        # Change kernel and min_noise to a list
        if type(kernel) not in (list,tuple): kernel = [kernel]*NbModel
        if type(min_noise) not in (list,tuple): min_noise = [min_noise]*NbModel
        models,likelihoods = [], []
        for i in range(NbModel):
            _output_scale = output_scale[:,i] if output_scale is not None else None
            likelihood, model = _SingleGPR(TrainIn, TrainOut[:,i], kernel[i], min_noise = min_noise[i],
                                            input_scale=input_scale, output_scale=_output_scale)
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

def _SingleGPR(TrainIn,TrainOut,kernel,prev_state=None,min_noise=None,input_scale=None,output_scale=None):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = Models.ExactGPmodel(TrainIn, TrainOut, likelihood, kernel)
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

    if input_scale is not None: model.input_scale = input_scale
    if output_scale is not None: model.output_scale = output_scale

    return likelihood, model


def _MultitaskGPR(TrainIn,TrainOut,kernel,prev_state=None,min_noise=None,input_scale=None,output_scale=None):
    ndim = TrainOut.shape[1]
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=ndim)
    model = Models.MultitaskGPModel(TrainIn, TrainOut, likelihood, kernel,rank=1)

    if input_scale is not None: model.input_scale = input_scale
    if output_scale is not None: model.output_scale = output_scale

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
    # print('\n################################\n')
    # print("Iterations: {}\nLoss: {}".format(i+1,TotalLoss))
    # print("\nModel parameters:")
    # PrintParameters(model)
    # print('################################\n')
    
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
    return np.mean(sqdiff)

def MAE(Predicted,Target):
    return np.abs(Predicted - Target).mean()/(Target.max() - Target.min())

def RMSE(Predicted,Target):
    return ((Predicted - Target)**2).mean()**0.5/(Target.max() - Target.min())

def Rsq(Predicted,Target):
    mean_pred = Predicted.mean()
    divisor = ((Predicted - mean_pred)**2).sum()
    MSE_val = ((Predicted - Target)**2).sum()
    return 1-(MSE_val/divisor)

def GetMetrics(model,x,target):
    with torch.no_grad():
        pred = model(x).mean.numpy()
    mse = MSE(pred,target)
    mae = MAE(pred,target)
    rmse = RMSE(pred,target)
    rsq = Rsq(pred,target)
    return mse,mae,rmse,rsq

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

def Writehdf(File, data_path, array, attrs={}):
    Database = Openhdf(File,'a')
    if data_path in Database:
        del Database[data_path]
    dset = Database.create_dataset(data_path,data=array)
    if attrs:
        dset.attrs.update(**attrs)
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

def GetMLdata2(DataFile_path,DataNames,ArrayName,Nb=-1):
    if type(DataNames)==str:DataNames = [DataNames]
    data = ["{}/{}".format(dataname,ArrayName) for dataname in DataNames]
    Data = Readhdf(DataFile_path,data)

    for i in range(len(DataNames)):
        _Nb = Nb[i] if type(Nb)==list else Nb
        if _Nb==-1:continue

        if type(_Nb)==int:
            Data[i] = Data[i][:_Nb]
        if type(_Nb) in (list,tuple):
            l,u = _Nb
            Data[i] = Data[i][l:u]

    return np.vstack(Data)

def GetMLattrs(DataFile_path,DataNames,ArrayName):
    if type(DataNames)==str:DataNames = [DataNames]
    Database = Openhdf(DataFile_path,'r')
    attrs = {}
    for dataname in DataNames:
        data_path = "{}/{}".format(dataname,ArrayName)
        data_attrs = Database[data_path].attrs
        attrs.update(**data_attrs)
    Database.close()
    return attrs

def WriteMLdata(DataFile_path,DataNames,ArrayName,DataList, attrs={}):
    for resname, data in zip(DataNames, DataList):
        DataPath = "{}/{}".format(resname,ArrayName) # path to data in file
        Writehdf(DataFile_path, DataPath, data, attrs=attrs)

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

def ModelSummary(NbInput,NbOutput,TrainNb,TestNb=None,Features=None,Labels=None):
    ModelDesc = "Model Summary\n\n"\
                "Nb.Inputs: {}\nNb.Outputs: {}\n\n"\
                "Nb.Train data: {}\nNb.Test data: {}\n\n".format(NbInput,NbOutput,TrainNb,TestNb)
    if Features is not None:
        if type(Features) != str: Features = ", ".join(Features)
        ModelDesc += "Input features:\n{}\n\n".format(Features)
    if Labels is not None:
        if type(Labels) != str: Labels = ", ".join(Labels)
        ModelDesc += "Output labels:\n{}\n".format(Labels)

    print(ModelDesc)
# ==============================================================================
# ML model Optima

def GetOptima(model, NbInit, bounds, seed=None, find='max', tol=0.01,
              order='decreasing', success_only=True, constraints=()):
    if seed!=None: np.random.seed(seed)
    init_points = np.random.uniform(0,1,size=(NbInit,len(bounds)))

    Optima = slsqp_multi(_GPR_Opt, init_points, bounds=bounds,
                         constraints=constraints,find=find, tol=tol,
                         order=order, success_only=success_only,
                         jac=True, args=[model])
    Optima_cd, Optima_val = Optima
    return Optima_cd, Optima_val

def GetExtrema(model,NbInit,bounds,seed=None):
    # ==========================================================================
    # Get min and max values for each
    Extrema_cd, Extrema_val = [], []
    for tp,order in zip(['min','max'],['increasing','decreasing']):
        _Extrema_cd, _Extrema_val = GetOptima(model, NbInit, bounds,seed,
                                              find=tp, order=order)
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

def InputQuery(model, NbInput, base=0.5, Ndisc=50):

    split = np.linspace(0,1,Ndisc+1)
    a = np.ones((Ndisc+1,NbInput))*base
    pred_mean,pred_std = [],[]
    for i in range(NbInput):
        _a = a.copy()
        _a[:,i] = split
        _a = torch.from_numpy(_a)
        with torch.no_grad():
            out = model.likelihood(model(_a))
            mean = out.mean.numpy()
            stdev = out.stddev.numpy()
        pred_mean.append(mean);pred_std.append(stdev)
    return pred_mean,pred_std



# ==============================================================================
def PCA(Data, metric={'threshold':0.99}):
    U,s,VT = np.linalg.svd(Data,full_matrices=False)

    if 'threshold' in metric:
        s_sc = np.cumsum(s)
        s_sc = s_sc/s_sc[-1]
        threshold_ix = np.argmax( s_sc > metric['threshold']) + 1
    else: threshold_ix = 0

    if 'error' in metric:
        for j in range(VT.shape[1]):
            Datacompress = Data.dot(VT[:j+1,:].T)
            Datauncompress = Datacompress.dot(VT[:j+1,:])
            diff = np.abs(Data - Datauncompress)/Data
            maxix = np.unravel_index(np.argmax(diff, axis=None), diff.shape)
            if diff[maxix] < metric['error']:
                error_ix = j
                break
    else: error_ix = 0

    ix = max(threshold_ix,error_ix)
    VT = VT[:ix,:]

    Datacompress = Data.dot(VT.T)
    Datauncompress = Datacompress.dot(VT)
    diff = np.abs(Data - Datauncompress)
    absmaxix = np.unravel_index(np.argmax(diff, axis=None), diff.shape)
    percmaxix = np.unravel_index(np.argmax(diff/Data, axis=None), diff.shape)

    print("Compressing data from {} to {} dimensions using PCA.\n"\
           "Max absolute error: Original={:.2f}, compressed={:.2f}\n"\
           "Max percentage error: Original={:.2f}, compressed={:.2f}\n"\
           .format(Data.shape[1],ix,Data[absmaxix],Datauncompress[absmaxix],
                   Data[percmaxix],Datauncompress[percmaxix])
          )

    return VT
