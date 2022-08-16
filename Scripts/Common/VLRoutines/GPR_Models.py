
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

    DataFile_path = "{}/{}".format(VL.PROJECT_DIR, Parameters.TrainData[0])
    TrainIn, TrainOut = ML.GetDataML(DataFile_path, *Parameters.TrainData[1:])

    DataFile_path = "{}/{}".format(VL.PROJECT_DIR, Parameters.TestData[0])
    TestIn, TestOut = ML.GetDataML(DataFile_path, *Parameters.TestData[1:])

    # ==========================================================================
    # Get parameters and build model
    ModelParameters = getattr(Parameters,'ModelParameters',{})
    TrainingParameters = getattr(Parameters,'TrainingParameters',{})
    likelihood, model, Dataspace = GPR.BuildModel([TrainIn,TrainOut],[TestIn,TestOut],
                            DADict['CALC_DIR'], ModelParameters=ModelParameters,
                            TrainingParameters=TrainingParameters)

    # ==========================================================================
    # Get performance metric of model
    Data = {'Train':[Dataspace.TrainIn_scale,Dataspace.TrainOut_scale],
            'Test':[Dataspace.TestIn_scale,Dataspace.TestOut_scale]}
    metrics_dict = Metrics(model,Data) # dict of pandas dfs with same keys as Data

    for key, val in metrics_dict.items():
        print('\n=============================================================\n')
        print('{} metrics'.format(key))
        print(val)
    print()

    NbOutput = TrainOut.shape[1] if TrainOut.ndim==2 else 1
    for i in range(NbOutput):
        print("Output_{}\n".format(i))

        for key,val in metrics_dict.items():
            d = ", ".join("{:.3e}".format(v) for v in val.iloc[i].tolist())
            print("{:<8}: {}".format(key,d))
        print()
        GPR.PrintParameters(model.models[i])



def GPR_PCA_hdf5(VL,DADict):

    Parameters = DADict['Parameters']

    NbTorchThread = getattr(Parameters,'NbTorchThread',None)
    if NbTorchThread: torch.set_num_threads(NbTorchThread)

    # ==========================================================================
    # Get Train & test data from file DataFile_path
    DataFile_path = "{}/{}".format(VL.PROJECT_DIR, Parameters.TrainData[0])
    TrainIn, TrainOut = ML.GetDataML(DataFile_path, *Parameters.TrainData[1:])

    DataFile_path = "{}/{}".format(VL.PROJECT_DIR, Parameters.TestData[0])
    TestIn, TestOut = ML.GetDataML(DataFile_path, *Parameters.TestData[1:])

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

    # ==========================================================================
    # Get performance metric of model
    Data = {'Train':[Dataspace.TrainIn_scale,Dataspace.TrainOut_scale],
            'Test':[Dataspace.TestIn_scale,Dataspace.TestOut_scale]}
    # dict of pandas dfs with same keys as Data
    metrics_dict,metrics_scaled_dict = Metrics_PCA(model,Data,VT, Dataspace.OutputScaler)

    for key, val in metrics_scaled_dict.items():
        print('\n=============================================================\n')
        print('{} metrics'.format(key))
        print(val)
    print()

    NbOutput = TrainOutCompress.shape[1]
    for i in range(NbOutput):
        print("Output_{}\n".format(i))

        for key,val in metrics_dict.items():
            d = ", ".join("{:.3e}".format(v) for v in val.iloc[i].tolist())
            print("{:<8}: {}".format(key,d))
        print()
        GPR.PrintParameters(model.models[i])

# ==============================================================================
# Functions used to asses performance of models

def Metrics(model, Data, fast_pred_var=True):
    # =========================================================================
    # Get error metrics for model
    metrics = {}
    for key, val in Data.items():
        data_in,data_out = val
        data_out = data_out.detach().numpy()

        pred_mean = _pred(model,data_in,fast_pred_var=fast_pred_var)
        df_data = ML.GetMetrics2(pred_mean,data_out)
        metrics[key] = df_data

    return metrics

def Metrics_PCA(model, Data, VT, OutputScaler, fast_pred_var=True):
    # =========================================================================
    # Get error metrics for model
    metrics,metrics_scaled = {}, {}
    for key, val in Data.items():
        data_in,data_out = val
        data_out = data_out.detach().numpy()

        pred_mean = _pred(model,data_in,fast_pred_var=fast_pred_var)
        df_data = ML.GetMetrics2(pred_mean,data_out)
        metrics[key] = df_data

        pred_mean_rescale = ML.DataRescale(pred_mean,*OutputScaler)
        data_out_rescale = ML.DataRescale(data_out,*OutputScaler)

        df_data_uncompress = ML.GetMetrics2(pred_mean_rescale.dot(VT),data_out_rescale.dot(VT))

        metrics_scaled[key] = df_data_uncompress.mean()

    return metrics, metrics_scaled

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
