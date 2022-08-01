
import os
import sys
import shutil

import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt

import VLFunctions as VLF
from Scripts.Common.tools import MEDtools
from Scripts.Common.ML import ML, Surrogate

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
    likelihood, model, Dataspace = BuildGPR([TrainIn,TrainOut],[TestIn,TestOut],
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
    likelihood, model, Dataspace = BuildGPR([TrainIn,TrainOutCompress],[TestIn,TestOutCompress],
                            DADict['CALC_DIR'], ModelParameters=ModelParameters,
                            TrainingParameters=TrainingParameters)

    # ==========================================================================
    # Get performance metric of model
    Metrics_PCA(model,[Dataspace.TrainIn_scale,Dataspace.TrainOut_scale],
            [Dataspace.TestIn_scale,Dataspace.TestOut_scale], VT, Dataspace.OutputScaler)

# ==============================================================================
# Generic building function for GPR
def BuildGPR(TrainData, TestData, ModelDir, ModelParameters={},
             TrainingParameters={}, FeatureNames=None,LabelNames=None):

    TrainIn,TrainOut = TrainData
    TestIn,TestOut = TestData

    Dataspace = ML.DataspaceTrain(TrainData,Test=TestData)

    # ==========================================================================
    # Model summary

    ML.ModelSummary(Dataspace.NbInput,Dataspace.NbOutput,Dataspace.NbTrain,
                    TestNb=TestIn.shape[0], Features=FeatureNames,
                    Labels=LabelNames)

    # ==========================================================================
    # get model & likelihoods
    likelihood, model = ML.Create_GPR(Dataspace.TrainIn_scale, Dataspace.TrainOut_scale,
                                   **ModelParameters,
                                   input_scale=Dataspace.InputScaler,
                                   output_scale=Dataspace.OutputScaler)

    # Train model
    Convergence = ML.GPR_Train(model, **TrainingParameters)
    model.eval()

    ModelSave(ModelDir,model,TrainIn,TrainOut,Convergence)

    return  likelihood, model, Dataspace

# ==============================================================================
# Functions used to asses performance of models

def Metrics(model, TrainData, TestData, fast_pred_var=True):
    # =========================================================================
    # Get error metrics for model

    for data,name in zip([TrainData,TestData],['Train','Test']):
        data_in,data_out = data
        data_out = data_out.detach().numpy()

        pred_mean = _pred(model,data_in,fast_pred_var=fast_pred_var)
        df_data = ML.GetMetrics2(pred_mean,data_out)

        print('\n=============================================================\n')
        print('{} metrics'.format(name))
        print(df_data)

def Metrics_PCA(model, TrainData, TestData, VT, OutputScaler, fast_pred_var=True):
    # =========================================================================
    # Get error metrics for model

    for data,name in zip([TrainData,TestData],['Train','Test']):
        data_in,data_out = data
        if hasattr(data_out,'detach'):
            data_out = data_out.detach().numpy()

        pred_mean = _pred(model,data_in,fast_pred_var=fast_pred_var)
        df_data = ML.GetMetrics2(pred_mean,data_out)

        print('\n=============================================================\n')
        print('{} metrics'.format(name))
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


# ==============================================================================
# Functions for saving & loading models
def ModelSave(ModelDir,model,TrainIn,TrainOut,Convergence):
    ''' Function to store model infromation'''
    # ==========================================================================
    # Save information
    os.makedirs(ModelDir,exist_ok=True)
    # save data
    np.save("{}/Input".format(ModelDir),TrainIn)
    np.save("{}/Output".format(ModelDir), TrainOut)

    # save model
    ModelFile = "{}/Model.pth".format(ModelDir)
    torch.save(model.state_dict(), ModelFile)

    # Plot convergence & save
    conv_len = [len(c) for c in Convergence]
    conv_sum = np.zeros(max(conv_len))
    for c in Convergence:
        conv_sum[:len(c)]+=np.array(c)
        conv_sum[len(c):]+=c[-1]

    plt.figure()
    plt.plot(conv_sum)
    plt.savefig("{}/Convergence.png".format(ModelDir),dpi=600)
    plt.close()

def Load_GPR(ModelDir):

    TrainIn = np.load("{}/Input.npy".format(ModelDir))
    TrainOut = np.load("{}/Output.npy".format(ModelDir))
    Dataspace = ML.DataspaceTrain([TrainIn,TrainOut])

    Parameters = VLF.ReadParameters("{}/Parameters.py".format(ModelDir))
    ModelParameters = getattr(Parameters,'ModelParameters',{})
    likelihood, model = ML.Create_GPR(Dataspace.TrainIn_scale, Dataspace.TrainOut_scale,
                                   input_scale=Dataspace.InputScaler,
                                   output_scale=Dataspace.OutputScaler,
                                   prev_state="{}/Model.pth".format(ModelDir),
                                   **ModelParameters)

    model.eval()

    return  likelihood, model, Dataspace, Parameters

def Load_GPR_PCA(ModelDir):

    VT = np.load("{}/VT.npy".format(ModelDir))
    ret = Load_GPR(ModelDir)

    return (*ret, VT)
