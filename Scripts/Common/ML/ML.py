import os
import sys
import time
from types import SimpleNamespace as Namespace

import numpy as np
import torch
import h5py
from natsort import natsorted
import pandas as pd

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

def ScaleValues(data,scaling='unit'):
    ''' '''
    if scaling.lower()=='unit':
        datamin,datamax = data.min(axis=0),data.max(axis=0)
        scaler = np.array([datamin,datamax-datamin])
    elif scaling.lower()=='centre':
        scaler = np.array([data.mean(axis=0),data.std(axis=0)])
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
def MSE(Predicted,Target,axis=-1,normalise=True):
    sqdiff = np.mean((Predicted - Target)**2,axis=axis)
    if normalise:
        sqdiff = sqdiff/(Target.max(axis=axis) - Target.min(axis=axis))
    return  sqdiff


def MAE(Predicted,Target,axis=None,normalise=True):
    mae = np.abs(Predicted - Target).mean(axis=axis)
    if normalise:
        mae = mae/(Target.max(axis=axis) - Target.min(axis=axis))
    return mae

def RMSE(Predicted,Target,axis=None,normalise=True):
    rmse = ((Predicted - Target)**2).mean(axis=axis)**0.5
    if normalise:
        rmse = rmse/(Target.max(axis=axis) - Target.min(axis=axis))
    return rmse

def Rsq(Predicted,Target,axis=None):
    mean_pred = Predicted.mean(axis=axis)
    divisor = ((Predicted - mean_pred)**2).sum(axis=axis)
    MSE_val = ((Predicted - Target)**2).sum(axis=axis)
    return 1-(MSE_val/divisor)

def GetMetrics2(pred,target):

    N = 1 if pred.ndim==1 else pred.shape[1]

    mse = MSE(pred,target,axis=0)
    mae = MAE(pred,target,axis=0)
    rmse = RMSE(pred,target,axis=0)
    rsq = Rsq(pred,target,axis=0)

    df=pd.DataFrame({"MSE":mse,"MAE":mae,"RMSE":rmse,"R^2":rsq},
                    index=["Output_{}".format(i) for i in range(N)])
    pd.options.display.float_format = '{:.3e}'.format
    return df


# ==============================================================================
# Functions used for reading & writing data

def Openhdf(File,style,timer=5):
    ''' Repeatedly attemps to open hdf file if it is held by another process for
    the time allocated by timer '''
    if style=='r' and not os.path.isfile(File):
        print(VLF.ErrorMessage("The following file does not exist:\n{}".format(File)))
        sys.exit()
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

def _Readhdf(File_handle, data_path,group=None):
    # add prefix to path
    if group: data_path = "{}/{}".format(group,data_path)
    # Check data is in file
    if data_path not in File_handle:
        print(VLF.ErrorMessage("data '{}' is not in file {}".format(data_path,File)))
        sys.exit()
    # Get data from file
    _data = File_handle[data_path][:]
    # Reshape 1D data to 2D
    # if _data.ndim==1:
    #     _data = _data.reshape((_data.size,1))
    return _data

def Readhdf(File, data_path, group=None):
    Database = Openhdf(File,'r')
    if type(data_path)==str:
        data = _Readhdf(Database,data_path,group)
    else:
        # assume its an iterable
        data = [_Readhdf(Database,_data_path,group) for _data_path in data_path]
    Database.close()
    return data

# ==============================================================================
# Functions used for ML work

def _GetData(DataFile,DataNames,group=None, Nb=-1):
    data = Readhdf(DataFile,DataNames,group=group) # read DataNames for DataFile

    for i in range(len(data)):
        _Nb = Nb[i] if type(Nb)==list else Nb
        if _Nb==-1:continue # -1 means we use all data

        if type(_Nb)==int:
            data[i] = data[i][:_Nb]
        if type(_Nb) in (list,tuple):
            l,u = _Nb
            data[i] = data[i][l:u]

    return data

def _GetDataStacked(DataFile,DataName,group=None, Nb=-1):
    data = _GetData(DataFile,DataName,group=group,Nb=Nb)
    if type(DataName)==list: # get a list back
        data = np.concatenate(data)
    return data

def GetData(DataFile,DataName,group=None, Nb=-1):
    if type(DataName)==list and type(DataName[0])==list:
        # multiple outputs which need to be stacked side by side
        data = []
        for _DataName in DataName:
            _data = _GetDataStacked(DataFile,_DataName,group=group,Nb=Nb) 
            if _data.ndim==1: 
                _data = _data.reshape(-1,1)
            data.append(_data)
        data = np.concatenate(data,axis=1)
    else:
        # DataName can be string or a list
        data = _GetDataStacked(DataFile,DataName,group=group,Nb=Nb)

    return data

def GetDataML(DataFile,InputName,OutputName,options={}):
    ''' This function gets inputs and outputs for supervised ML. '''
    in_data = GetData(DataFile,InputName,**options)
    out_data = GetData(DataFile,OutputName,**options)
    return in_data, out_data

def VLGetDataML(VL,Data):

    DataFile_path = "{}/{}".format(VL.PROJECT_DIR, Data[0])
    return GetDataML(DataFile_path, *Data[1:])

# ==============================================================================
# Functions used to gather data
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

def ModelSummary(NbInput,NbOutput,TrainNb,**kwargs):
    ModelDesc = "Model Summary\n\n"\
                "Nb.Inputs: {}\nNb.Outputs: {}\n"\
                "Nb.Train data: {}\n".format(NbInput,NbOutput,TrainNb)

    for name, value in kwargs.items():
        if value is None: continue
        ModelDesc+="{}: {}\n".format(name,value)

    print(ModelDesc)

# ==============================================================================
# ML model Optima

def GetOptima(model, NbInit, bounds, seed=None, find='max', tol=0.01,
              order='decreasing', success_only=True, constraints=(),fnc=None,fnc_args=None):
    if seed!=None: np.random.seed(seed)
    init_points = np.random.uniform(0,1,size=(NbInit,len(bounds)))

    # go to default function
    if fnc is None:
        fnc = _model_opt
        fnc_args = [model]

    Optima = slsqp_multi(fnc, init_points, bounds=bounds,
                         constraints=constraints,find=find, tol=tol,
                         order=order, success_only=success_only,
                         jac=True, args=fnc_args)
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

def _model_opt(X,model):
    torch.set_default_dtype(torch.float64)
    X = torch.tensor(X)
    output_val,output_grad = model.OptimiseOutput(X)
    return output_val.detach().numpy(), output_grad.detach().numpy()

def temporary_numpy_seed(fnc,seed=None,args=()):
    ''' Function which will temporary set the numpy random seed to seed
        and peform a function evaluation.'''
    if seed is None:
        # no seeding performed
        out = fnc(*args)
    else:
        st0 = np.random.get_state() # get current state
        np.random.seed(seed) # set new random state
        out = fnc(*args) # make function call
        np.random.set_state(st0) # reset to original state

    return out




def _init_points(nb_points,bounds):
    # checks
    points = [np.random.uniform(*bound,nb_points) for bound in bounds]
    return np.transpose(points)


def GetRandomPoints(nb_points,bounds,seed=None):
    ''' nb_points number of points are randomly drawn from the
        hyper space defined by bounds'''

    random_points = temporary_numpy_seed(_init_points,seed=seed,args=(nb_points,bounds))
    return random_points

def Optimise(fnc, NbInit, bounds, fnc_args=(), seed=None, find='max', tol=0.01,
              order='decreasing', success_only=True,**kwargs):

    init_points = GetRandomPoints(NbInit,bounds,seed=seed)
    Optima_cd, Optima_val = slsqp_multi(fnc, init_points,
                             bounds=bounds, find=find, tol=tol, order=order,
                             success_only=success_only,jac=True, args=fnc_args,
                             **kwargs)
    return Optima_cd, Optima_val


def OptimiseLSE(fnc, target, NbInit, bounds, fnc_args=(), filter=True, scale_factor=1,find='min', **kwargs):
    lse_fnc_args = [target,fnc,fnc_args,scale_factor]
    cd,val = Optimise(_LSE_func, NbInit, bounds, fnc_args=lse_fnc_args, find=find, **kwargs)

    if filter:
        pred = fnc(cd,*fnc_args)[0] # no need for gradient
        keep = LSE_filter(pred,target,err=0.05)
        return cd[keep], val[keep]
    else:
        return cd, val

def _LSE_func(X,target,fnc,fnc_args,scale_factor=1):
    ''' Function for solving least squares error optimisation.
        Use scale factor as graidents can be very large, leading to slow convergence time'''
    pred, grad = fnc(X,*fnc_args) # get the value and gradient from the given function
    diff = pred - target
    lse_pred = (diff**2)
    lse_grad = 2*(diff.T*grad.T).T
    if lse_pred.ndim==2:
        lse_pred = lse_pred.mean(axis=1)
        lse_grad = lse_grad.mean(axis=1)
    return lse_pred/scale_factor, lse_grad/scale_factor

def LSE_filter(pred,target,err=0.05):
    abs_err = np.abs(pred - target)/target
    keep =  (abs_err < err)
    if pred.ndim==2: keep = keep.all(axis=1)
    return keep



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

def PCA(Data,centre=True):
    ''' Data should be centred around origin and equally scaled for best performance.
        SVD of original matric is more robust than PCA of covariance.'''

    if centre:
        print('Centering data to 0 mean and std.dev. of 1')
        ScalePCA = ScaleValues(Data,'centre')
        Data = DataScale(Data,*ScalePCA)

    U,s,VT = np.linalg.svd(Data,full_matrices=False)
    return U,s,VT


def PCA_threshold(eigens,threshold):
    eigens_cs = np.cumsum(eigens)
    eigens_cs = eigens_cs/eigens_cs[-1] # scaled cumulative sum
    threshold_ix = np.argmax( eigens_cs >= threshold) + 1
    return threshold_ix

def GetPC(Data,metric={},centre=True,check_variance=False):
    _data = np.copy(Data)
    U,s,VT = PCA(Data,centre=centre)

    # get eigenvalues from s. This formula only holds if the data is centred.
    eigens = s**2/(Data.shape[0] - 1)

    if check_variance:
        PC_var = eigens.sum()
        Data_var = np.var(_data,axis=0).sum()*Data.shape[0]/(Data.shape[0] - 1)
        print("Original variance in data:{}\nVariance on principal components:{}".format(Data_var,PC_var))

    if 'threshold' in metric:
        threshold_ix = PCA_threshold(eigens,metric['threshold'])
    else: threshold_ix=0

    # number of principal components to keep
    component_ix = int(metric['nb_components']) if 'nb_components' in metric else 0

    max_ix = max([threshold_ix,component_ix])

    if max_ix>0: keep_ix=max_ix # keep best
    else: keep_ix=len(s) # use full rank of PCs

    U,s,VT =U[:,:keep_ix],s[:keep_ix], VT[:keep_ix,:] # only keep the first ix eigenvectors for compression
    return U,s,VT

def PCA_track(TrainData,TestData,centre=True):

    if centre:
        ScalePCA = ScaleValues(TrainData,'centre')
        TrainData = DataScale(TrainData,*ScalePCA)
        TestData = DataScale(TestData,*ScalePCA)

    U,s,VT = PCA(TrainData,centre=False)

    Traindata_compress = TrainData.dot(VT.T)
    Testdata_compress = TestData.dot(VT.T)

    train_recon,test_recon = np.zeros(TrainData.shape),np.zeros(TestData.shape)
    train_rmse,test_rmse = [],[]
    for i in range(VT.shape[0]):
        train_recon += Traindata_compress[:,i:i+1].dot(VT[i:i+1])
        _train_rmse = RMSE(train_recon,TrainData, axis=0).mean()
        train_rmse.append(_train_rmse)

        test_recon += Testdata_compress[:,i:i+1].dot(VT[i:i+1])
        _test_rmse = RMSE(test_recon,TestData, axis=0).mean()
        test_rmse.append(_test_rmse)

    return train_rmse,test_rmse


def PCA_check_convergence(loss_data,convergence=0.99,nb_convergence=3):
    loss_data = np.array(loss_data)
    frac = loss_data[1:]/loss_data[:-1]

    for j in range(len(frac) - nb_convergence):
        check_conv = frac[j:j+nb_convergence]
        if (np.array(check_conv)>convergence).all():
            break
    return j

def PCA_convergence(TrainData,TestData,centre=True):
    train_rmse,test_rmse = PCA_track(TrainData,TestData,centre=centre)
    convergence_ix = _PCA_check_convergence(test_rmse)
    return convergence_ix


# ==============================================================================
# Clustering

def Inertia(data,data_cluster):
    unique = np.unique(data_cluster,axis=0)
    inertia = 0
    for val in unique:
        ixs = np.where(data_cluster==val)[0]
        inertia += ((data[ixs] - val)**2).sum()
    return inertia

def GVF(data,data_cluster):
    # Squared deviation for mean array
    SDAM = ((data - data.mean(axis=0))**2).sum()
    # Squared deviation class mean
    SDCM = Inertia(data,data_cluster)
    return (SDAM-SDCM)/SDAM

def Kmeans(data, nb_cluster, n_init=3, seed=None):
    from sklearn.cluster import KMeans

    if seed is not None:
        state = np.random.get_state() # get current random state
        np.random.seed(seed) # set random state to seed value for reproducability
    if nb_cluster >= data.shape[0]:
        print('Number of clusters is greater than number of samples')
        return data

    if data.ndim==1: _data = data.reshape(-1,1)
    else: _data = data

    ScaleValue = ScaleValues(_data)
    _data = DataScale(_data,*ScaleValue)

    # ==========================================================================
    # cluster JH_Vol in to nb_cluster groups
    kmeans = KMeans(n_clusters=nb_cluster,n_init=n_init,n_jobs=1).fit(_data)

    # ==========================================================================
    # calculate goodness of fit
    SDAM = ((_data - _data.mean(axis=0))**2).sum() # Squared deviation for mean array
    SDCM = kmeans.inertia_ # Squared deviation class mean
    GFV = (SDAM-SDCM)/SDAM # Goodness of fit value

    # ==========================================================================
    # rescale JH_vol to original range. this will have nb_cluster entries
    cluster_centres = DataRescale(kmeans.cluster_centers_,*ScaleValue)

    data_cluster = np.zeros(data.shape)
    for i,cluster_val in enumerate(cluster_centres):
        data_cluster[kmeans.labels_==i] = cluster_val

    if seed is not None:
        np.random.set_state(state)

    return data_cluster

# ==============================================================================
# Model wrappers

class ModelWrapBase():
    '''
    Class which pulls together useful functions which makes using pytorch's ML
    models easier to use. This class is not designed to be used as is, but should
    be added to depending on the model type.
    '''

    def __init__(self,model,Dataspace):
        self.model = model
        self.Dataspace = Dataspace

    def _get_dset(self,dset_name):
        if not hasattr(self.Dataspace,dset_name):
            print('Error: {} is not associated with the dataspace'.format(dset_name))
        else:
            return getattr(self.Dataspace,dset_name)

    def GetTrainData(self,to_numpy=False,scale=True):
        return self.GetDataset('Train',to_numpy=to_numpy,scale=scale)

    def GetDataset(self,dset_name,to_numpy=False,scale=True):
        return self.GetDatasetInput(dset_name,to_numpy=to_numpy,scale=scale), self.GetDatasetOutput(dset_name,to_numpy=to_numpy,scale=scale)

    def GetDatasetInput(self,dset_name,to_numpy=False,scale=True):
        input = self._get_dset("{}In_scale".format(dset_name))
        if to_numpy:
            input = input.detach().numpy()
        if scale:
            input = self.RescaleInput(input)
        return input

    def GetDatasetOutput(self,dset_name,to_numpy=False, scale=True):
        output = self._get_dset("{}Out_scale".format(dset_name))
        if to_numpy:
            output = output.detach().numpy()
        if scale:
            output = self.RescaleOutput(output)            
        return output

    def AddDataset(self, data, dset_name):
        DataspaceAdd(self.Dataspace,**{dset_name:data})

    def ScaleInput(self,input):
        return DataScale(input,*self.Dataspace.InputScaler)

    def RescaleInput(self,input):
        return DataRescale(input,*self.Dataspace.InputScaler)

    def ScaleOutput(self,output):
        return DataScale(output,*self.Dataspace.OutputScaler)

    def RescaleOutput(self,output):
        return DataRescale(output,*self.Dataspace.OutputScaler)

    def Predict_dset(self,dset_name, scale_outputs=True):
        inputs = self.GetDatasetInput(dset_name)
        return self.Predict(inputs,scale_outputs=scale_outputs)
