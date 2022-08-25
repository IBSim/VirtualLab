import os
import sys
import time
import pandas as pd
from types import SimpleNamespace as Namespace

import numpy as np
import torch
import h5py
from natsort import natsorted
import time

from Scripts.Common.Optimisation import slsqp_multi
from Scripts.Common import VLFunctions as VLF
from Scripts.Common.tools import Paralleliser

# ==============================================================================
# Routine for storing data and scaling
def DataspaceAdd(Dataspace,**kwargs):
    for varname, data in kwargs.items():
        data_in, data_out = data
        scaled_in = DataScale(data_in,*Dataspace.InputScaler)
        setattr(Dataspace,"{}In_scale".format(varname), torch.from_numpy(scaled_in))
        scaled_out = DataScale(data_out,*Dataspace.OutputScaler)
        setattr(Dataspace,"{}Out_scale".format(varname), torch.from_numpy(scaled_out))

def DataspaceTrain(TrainData, **kwargs):

    TrainIn,TrainOut = TrainData

    # ==========================================================================
    # Scale ranges for training data
    InputScaler = ScaleValues(TrainIn)
    OutputScaler = ScaleValues(TrainOut)
    # scale training data
    TrainIn_scale = DataScale(TrainIn,*InputScaler)
    TrainOut_scale = DataScale(TrainOut,*OutputScaler)
    # convert to tensors
    TrainIn_scale = torch.from_numpy(TrainIn_scale)
    TrainOut_scale = torch.from_numpy(TrainOut_scale)

    NbInput = TrainIn.shape[1] if TrainIn.ndim==2 else 1
    NbOutput = TrainOut.shape[1] if TrainOut.ndim==2 else 1

    Dataspace = Namespace(InputScaler=InputScaler,OutputScaler=OutputScaler,
                    NbInput=NbInput, NbOutput=NbOutput,
                    NbTrain=TrainIn.shape[0], TrainIn_scale=TrainIn_scale,
                    TrainOut_scale=TrainOut_scale)

    DataspaceAdd(Dataspace,**kwargs)

    return Dataspace

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

def GetMetrics2(pred,target):

    N = 1 if pred.ndim==1 else pred.shape[1]

    mse = MSE(pred,target)
    mae = MAE(pred,target)
    rmse = RMSE(pred,target)
    rsq = Rsq(pred,target)


    df=pd.DataFrame({"MSE":mse,"MAE":mae,"RMSE":rmse,"R^2":rsq},
                    index=["Output_{}".format(i) for i in range(N)])
    pd.options.display.float_format = '{:.3e}'.format
    return df


# ==============================================================================
# Functions used for reading & writing data

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

def Writehdf(File, data_path, array, attrs={}, group=None):
    if group: data_path = "{}/{}".format(group,data_path)

    Database = Openhdf(File,'a')
    if data_path in Database:
        del Database[data_path]
    dset = Database.create_dataset(data_path,data=array)
    if attrs:
        dset.attrs.update(**attrs)
    Database.close()

def Readhdf(File, data_paths,group=None):
    Database = Openhdf(File,'r')
    if type(data_paths)==str: data_paths = [data_paths]
    data = []
    for data_path in data_paths:
        # add prefix to pat
        if group: data_path = "{}/{}".format(group,data_path)
        # Check data is in file
        if data_path not in Database:
            print(VLF.ErrorMessage("data '{}' is not in file {}".format(data_path,File)))
            sys.exit()
        # Get data from file
        _data = Database[data_path][:]
        # Reshape 1D data to 2D
        if _data.ndim==1:
            _data = _data.reshape((_data.size,1))

        data.append(_data)

    Database.close()

    return data

# ==============================================================================
# Functions used for ML work

def GetData(DataFile,DataNames,group=None, Nb=-1):
    data = Readhdf(DataFile,DataNames,group=group) # read DataNames for DataFile

    for i in range(len(data)):
        _Nb = Nb[i] if type(Nb)==list else Nb
        if _Nb==-1:continue # -1 means we use all data

        if type(_Nb)==int:
            data[i] = data[i][:_Nb]
        if type(_Nb) in (list,tuple):
            l,u = _Nb
            data[i] = data[i][l:u]

    return np.vstack(data)

def GetDataML(DataFile,InputNames,OutputNames,options={}):
    ''' This function gets inputs and outputs for supervised ML. '''
    in_data = GetData(DataFile,InputNames,**options)
    out_data = GetData(DataFile,OutputNames,**options)
    return in_data, out_data

def GetResPaths(ResDir,DirOnly=True,Skip=['_']):
    ''' This returns a naturally sorted list of the directories in ResDir'''
    ResPaths = []
    for _dir in natsorted(os.listdir(ResDir)):
        if _dir.startswith(tuple(Skip)): continue
        path = "{}/{}".format(ResDir,_dir)
        if DirOnly and os.path.isdir(path):
            ResPaths.append(path)

    return ResPaths

def ExtractData(ResPath,functions,args,kwargs):
    ''' Function which extracts data from results directory ResPath using functions,
        args and kwargs. '''
    ret = []
    for _function,_args,_kwargs in zip(functions,args,kwargs):
        _ret = _function(ResPath,*_args,**_kwargs)
        ret.append(_ret)
    return ret

def ExtractData_Dir(ResDir,functions,args,kwargs, parallel_options={}):
    ''' Parallelised function which allows data to be extracted from all
        results in a directory. M sets of data can be extracted using the M
        functions, args and kwargs for the N results in ResDir. '''

    # Get paths to all directories in ResDir
    ResPaths = GetResPaths(ResDir)
    # write args in format so that it can be passed to the paralleliser function
    Args_parallel = [[p,functions,args,kwargs] for p in ResPaths]
    # Pass function and arguments to paralleliser (default behaviour: no parallelisation)
    Res = Paralleliser(ExtractData,Args_parallel,**parallel_options)
    # re-order results Res from N lists of length M to M lists of length N
    Res = list(zip(*Res))
    return Res

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
# Data compression algorithms

def PCA(Data, metric={}):
    ''' Funtion which performs principal component analysis'''
    U,s,VT = np.linalg.svd(Data,full_matrices=False)
    nb_component = len(s)

    # threshold of data variation to get above
    if 'threshold' in metric:
        s_sc = np.cumsum(s)
        s_sc = s_sc/s_sc[-1]
        threshold_ix = np.argmax( s_sc >= metric['threshold']) + 1
    else: threshold_ix = 0

    # error % to get below
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

    # number of principal components to keep
    component_ix = int(metric['components'])if 'components' in metric else 0

    ix = max(threshold_ix,error_ix,component_ix)
    if ix==0: ix = nb_component

    VT = VT[:ix,:] # only keep the first ix eigenvectors for compression
    return VT
