
import os
import sys
import shutil
import pandas as pd

import numpy as np
import torch
import gpytorch

from Scripts.Common.tools import MEDtools
from Scripts.Common.ML import ML, GPR

dtype = 'float64' # float64 is more accurate for optimisation purposes
torch_dtype = getattr(torch,dtype)
torch.set_default_dtype(torch_dtype)

# ==============================================================================
# VirtualLab compatible models

def GPR_hdf5(VL,DADict):
    Parameters = DADict['Parameters']

    NbTorchThread = getattr(Parameters,'NbTorchThread',None)
    if NbTorchThread: torch.set_num_threads(NbTorchThread)

    # ==========================================================================
    # Get Train & test data from file DataFile_path
    DataFile_path = "{}/{}".format(VL.PROJECT_DIR, Parameters.DataFile)
    TrainIn, TrainOut = ML.GetMLdata(DataFile_path, Parameters.TrainData,
                                     Parameters.InputArray, Parameters.OutputArray,
                                     getattr(Parameters,'TrainNb',-1))
    TestIn, TestOut = ML.GetMLdata(DataFile_path, Parameters.TestData,
                                   Parameters.InputArray, Parameters.OutputArray,
                                   getattr(Parameters,'TestNb',-1))

    # ==========================================================================
    # Get parameters and build model
    ModelParameters = getattr(Parameters,'ModelParameters',{})
    TrainingParameters = getattr(Parameters,'TrainingParameters',{})
    likelihood, model, Dataspace = GPR.BuildModel([TrainIn,TrainOut],[TestIn,TestOut],
                            DADict['CALC_DIR'], ModelParameters=ModelParameters,
                            TrainingParameters=TrainingParameters)

    # ==========================================================================
    # Get performance metric of model
    Metrics(model,[Dataspace.TrainIn_scale,Dataspace.TrainOut_scale],
            [Dataspace.TestIn_scale,Dataspace.TestOut_scale])

def GPR_PCA_hdf5(VL,DADict):
    # np.random.seed(100)
    Parameters = DADict['Parameters']

    NbTorchThread = getattr(Parameters,'NbTorchThread',None)
    if NbTorchThread: torch.set_num_threads(NbTorchThread)

    # ==========================================================================
    # Get Train & test data from file DataFile_path
    DataFile_path = "{}/{}".format(VL.PROJECT_DIR, Parameters.DataFile)
    TrainIn, TrainOut = ML.GetMLdata(DataFile_path, Parameters.TrainData,
                                     Parameters.InputArray, Parameters.OutputArray,
                                     getattr(Parameters,'TrainNb',-1))
    TestIn, TestOut = ML.GetMLdata(DataFile_path, Parameters.TestData,
                                   Parameters.InputArray, Parameters.OutputArray,
                                   getattr(Parameters,'TestNb',-1))
    # ==========================================================================
    # Compress data & save compression matrix in CALC_DIR
    if os.path.isfile("{}/VT.npy".format(DADict['CALC_DIR'])) and not getattr(Parameters,'VT',True):
        VT = np.load("{}/VT.npy".format(DADict['CALC_DIR']))
    else:
        VT = ML.PCA(TrainOut,metric=getattr(Parameters,'Metric',{'threshold':0.99}))
        np.save("{}/VT.npy".format(DADict['CALC_DIR']),VT)

    TrainOutCompress = TrainOut.dot(VT.T)
    TestOutCompress = TestOut.dot(VT.T)

    # ==========================================================================
    # Get parameters and build model
    ModelParameters = getattr(Parameters,'ModelParameters',{})
    TrainingParameters = getattr(Parameters,'TrainingParameters',{})
    likelihood, model, Dataspace = GPR.BuildModel([TrainIn,TrainOutCompress],[TestIn,TestOutCompress],
                            DADict['CALC_DIR'], ModelParameters=ModelParameters,
                            TrainingParameters=TrainingParameters)

    # ==========================================================================
    # Get performance metric of model
    Metrics_PCA(model,[Dataspace.TrainIn_scale,Dataspace.TrainOut_scale],
            [Dataspace.TestIn_scale,Dataspace.TestOut_scale], VT, Dataspace.OutputScaler)

# ==============================================================================
# Functions used to asses performance of models

def Metrics(model, TrainData, TestData, fast_pred_var=True):
    # =========================================================================
    # Get error metrics for model

    for data,name in zip([TrainData,TestData],['Train','Test']):
        print('\n=============================================================\n')
        print('{} metrics'.format(name))

        data_in,data_out = data
        data_out = data_out.detach().numpy()

        pred_mean = _pred(model,data_in,fast_pred_var=fast_pred_var)
        df_data = ML.GetMetrics2(pred_mean,data_out)

        with pd.option_context('display.max_rows', None):
            print(df_data)

def Metrics_PCA(model, TrainData, TestData, VT, OutputScaler, fast_pred_var=True):
    # =========================================================================
    # Get error metrics for model

    for data,name in zip([TrainData,TestData],['Train','Test']):
        print('\n=============================================================\n')
        print('{} metrics'.format(name))
        data_in,data_out = data
        if hasattr(data_out,'detach'):
            data_out = data_out.detach().numpy()

        pred_mean = _pred(model,data_in,fast_pred_var=fast_pred_var)
        df_data = ML.GetMetrics2(pred_mean,data_out)

        with pd.option_context('display.max_rows', None):
            print(df_data)

        pred_mean_rescale = ML.DataRescale(pred_mean,*OutputScaler)
        data_out_rescale = ML.DataRescale(data_out,*OutputScaler)

        df_data_uncompress = ML.GetMetrics2(pred_mean_rescale.dot(VT),data_out_rescale.dot(VT))
        print('\nUncompresses output (averaged)')
        print(df_data_uncompress.mean())

def _pred(model,input,fast_pred_var=True):
    def _predfn(model,input):
        if hasattr(model,'models'):
            pred = model(*[input]*len(model.models))
            pred_mean = np.transpose([p.mean.numpy() for p in pred])
        else:
            pred_mean = model(input).mean.numpy()
        return pred_mean

    if fast_pred_var:
        with torch.no_grad(),gpytorch.settings.fast_pred_var():
            pred_mean = _predfn(model,input)
    else:
        with torch.no_grad():
            pred_mean = _predfn(model,input)

    return pred_mean
