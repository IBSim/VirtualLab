
import os
import sys
import shutil
import pandas as pd

import numpy as np
import torch
import gpytorch

from Scripts.Common.tools import MEDtools
from Scripts.Common.ML import ML, GPR, Adaptive

dtype = 'float64' # float64 is more accurate for optimisation purposes
torch_dtype = getattr(torch,dtype)
torch.set_default_dtype(torch_dtype)

# ==============================================================================
# VirtualLab compatible models

def GPR_data(VL,DataDict):
    Parameters = DataDict['Parameters']

    NbTorchThread = getattr(Parameters,'NbTorchThread',None)
    if NbTorchThread: torch.set_num_threads(NbTorchThread)

    TrainInput = np.array(Parameters.TrainInputData)
    TrainOutput = np.array(Parameters.TrainOutputData)

    # if TrainOutput.ndim==2 and TrainOutput.shape[1]==1:
    #     TrainOut = TrainOut.flatten()

    # ==========================================================================
    # Get parameters and build model
    ModelParameters = getattr(Parameters,'ModelParameters',{})
    TrainingParameters = getattr(Parameters,'TrainingParameters',{})
    likelihood, model, Dataspace = GPR.BuildModel([TrainInput,TrainOutput],
                            DataDict['CALC_DIR'], # where model will be saved to
                            ModelParameters=ModelParameters,
                            TrainingParameters=TrainingParameters)

    # add Test data to Dataspace (where it is scaled using training data)

    # TestInput = Parameters.TestInputData
    # TestOutput = Parameters.TestOutputData
    # ML.DataspaceAdd(Dataspace,Test=[TestInput,TestOutput])

    # ==========================================================================
    # Get performance metric of model
    Data = {'Train':[Dataspace.TrainIn_scale,Dataspace.TrainOut_scale],
     }# 'Test':[Dataspace.TestIn_scale,Dataspace.TestOut_scale]}

    Performance(model, Data, getattr(Parameters,'PrintParameters',False))


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
    if TrainOut.ndim==2 and TrainOut.shape[1]==1:
        TrainOut,TestOut = TrainOut.flatten(),TestOut.flatten()

    # ==========================================================================
    # Get parameters and build model
    ModelParameters = getattr(Parameters,'ModelParameters',{})
    TrainingParameters = getattr(Parameters,'TrainingParameters',{})
    likelihood, model, Dataspace = GPR.BuildModel([TrainIn,TrainOut],
                            DADict['CALC_DIR'], ModelParameters=ModelParameters,
                            TrainingParameters=TrainingParameters)

    # add Test data to Dataspace (where it is scaled using training data)
    ML.DataspaceAdd(Dataspace,Test=[TestIn,TestOut])

    # ==========================================================================
    # Get performance metric of model
    Data = {'Train':[Dataspace.TrainIn_scale,Dataspace.TrainOut_scale],
            'Test':[Dataspace.TestIn_scale,Dataspace.TestOut_scale]}

    Performance(model, Data, getattr(Parameters,'PrintParameters',False))

def GPR_hdf5_Metrics(VL,DADict):
    Parameters = DADict['Parameters']

    # ==========================================================================
    # Load model
    ModelDir = "{}/{}".format(VL.PROJECT_DIR,Parameters.ModelDir)
    likelihood, model, Dataspace, ParametersMod = GPR.LoadModel(ModelDir)

    # ==========================================================================
    # Get Test data from file DataFile_path
    DataFile_path = "{}/{}".format(VL.PROJECT_DIR, Parameters.TestData[0])
    TestIn, TestOut = ML.GetDataML(DataFile_path, *Parameters.TestData[1:])

    if TestOut.ndim==2 and TestOut.shape[1]==1:
        TestOut = TestOut.flatten()
    ML.DataspaceAdd(Dataspace,Test=[TestIn,TestOut])

    # ==========================================================================
    # Get performance metric of model
    Data = {'Train':[Dataspace.TrainIn_scale,Dataspace.TrainOut_scale],
            'Test':[Dataspace.TestIn_scale,Dataspace.TestOut_scale]}
    Performance(model, Data, getattr(Parameters,'PrintParameters',False))

def GPR_Adaptive(VL,DADict):
    Parameters = DADict['Parameters']

    # ==========================================================================
    # Load model
    ModelDir = "{}/{}".format(VL.PROJECT_DIR,Parameters.ModelDir)
    likelihood, model, Dataspace, ParametersMod = GPR.LoadModel(ModelDir)

    # ==========================================================================
    # Get next points to collect data
    bounds = [[0,1]]*Dataspace.NbInput

    BestPoints = Adaptive.Adaptive(model, Parameters.Adaptive, bounds, Show=5)
    BestPoints = ML.DataRescale(np.array(BestPoints),*Dataspace.InputScaler)
    print(BestPoints)
    # DADict['Data']['BestPoints'] = BestPoints

def GPR_PCA_hdf5(VL,DADict):

    Parameters = DADict['Parameters']

    NbTorchThread = getattr(Parameters,'NbTorchThread',None)
    if NbTorchThread: torch.set_num_threads(NbTorchThread)

    # ==========================================================================
    # Get Train & test data from file DataFile_path
    DataFile_path = "{}/{}".format(VL.PROJECT_DIR, Parameters.TrainData[0])
    TrainIn, TrainOut = ML.GetDataML(DataFile_path, *Parameters.TrainData[1:])
    ScalePCA = ML.ScaleValues(TrainOut,scaling='centre')
    TrainOut_centre = ML.DataScale(TrainOut,*ScalePCA)

    DataFile_path = "{}/{}".format(VL.PROJECT_DIR, Parameters.TestData[0])
    TestIn, TestOut = ML.GetDataML(DataFile_path, *Parameters.TestData[1:])
    TestOut_centre = ML.DataScale(TestOut,*ScalePCA)

    np.save("{}/ScalePCA.npy".format(DADict['CALC_DIR']),ScalePCA)

    # ==========================================================================
    # Compress data & save compression matrix in CALC_DIR
    if os.path.isfile("{}/VT.npy".format(DADict['CALC_DIR'])) and not getattr(Parameters,'VT',True):
        VT = np.load("{}/VT.npy".format(DADict['CALC_DIR']))
    else:
        metric = getattr(Parameters,'Metric',{})
        U,s,VT = ML.GetPC(TrainOut_centre,metric=metric,centre=False)# no need to centre as already done
        np.save("{}/VT.npy".format(DADict['CALC_DIR']),VT)

    TrainOutCompress = TrainOut_centre.dot(VT.T)
    TestOutCompress = TestOut_centre.dot(VT.T)


    NbComponents = VT.shape[0]
    print("Nb Components: {}".format(NbComponents))
    for name,orig,compress in zip(['Train','Test'],[TrainOut,TestOut],[TrainOutCompress,TestOutCompress]):
        reconstructed = ML.DataRescale(compress.dot(VT),*ScalePCA)
        diff = reconstructed - orig # compare uncompressed and original
        absmaxix = np.unravel_index(np.argmax(np.abs(diff), axis=None), diff.shape)
        percmaxix = np.unravel_index(np.argmax(np.abs(diff)/orig, axis=None), diff.shape)
        abs_orig, perc_orig = orig[absmaxix],orig[percmaxix]
        abs_uc, perc_uc = abs_orig+diff[absmaxix], perc_orig+diff[percmaxix]
        print('{} data compression\n'\
              'Max. abs. error: {:.3e} ({:.3e} v {:.3e})\n'\
              'Max. % error: {:.2f} ({:.3e} v {:.3e})\n'\
              .format(name,abs_uc - abs_orig,abs_uc,abs_orig,
                        100*(perc_uc - perc_orig)/perc_orig,perc_uc, perc_orig))

    # ==========================================================================
    # Get parameters and build model
    ModelParameters = getattr(Parameters,'ModelParameters',{})
    TrainingParameters = getattr(Parameters,'TrainingParameters',{})
    likelihood, model, Dataspace = GPR.BuildModel([TrainIn,TrainOutCompress],
                            DADict['CALC_DIR'], ModelParameters=ModelParameters,
                            TrainingParameters=TrainingParameters)

    # add Test data to Dataspace (where it is scaled using training data)
    ML.DataspaceAdd(Dataspace,Test=[TestIn,TestOutCompress])

    # ==========================================================================
    # Get performance metric of model
    Data = {'Train':[Dataspace.TrainIn_scale,Dataspace.TrainOut_scale],
            'Test':[Dataspace.TestIn_scale,Dataspace.TestOut_scale]}
    PrintParameters = getattr(Parameters,'PrintParameters',False)
    Performance(model, Data, PrintParameters=PrintParameters)
    Performance_PCA(model, Data, VT,Dataspace.OutputScaler, PrintParameters=PrintParameters)

def GPR_PCA_hdf5_Metrics(VL,DADict):
    Parameters = DADict['Parameters']

    # ==========================================================================
    # Load model
    ModelDir = "{}/{}".format(VL.PROJECT_DIR,Parameters.ModelDir)
    likelihood, model, Dataspace, ParametersMod = GPR.LoadModel(ModelDir)
    VT = np.load("{}/VT.npy".format(ModelDir))
    ScalePCA = np.load("{}/ScalePCA.npy".format(ModelDir))

    # ==========================================================================
    # Get Test data from file DataFile_path
    DataFile_path = "{}/{}".format(VL.PROJECT_DIR, Parameters.TestData[0])
    TestIn, TestOut = ML.GetDataML(DataFile_path, *Parameters.TestData[1:])
    TestOut_centre = ML.DataScale(TestOut,*ScalePCA)
    TestOutCompress = TestOut_centre.dot(VT.T)

    ML.DataspaceAdd(Dataspace,Test=[TestIn,TestOutCompress])

    # ==========================================================================
    # Get performance metric of model
    Data = {'Train':[Dataspace.TrainIn_scale,Dataspace.TrainOut_scale],
            'Test':[Dataspace.TestIn_scale,Dataspace.TestOut_scale]}
    PrintParameters = getattr(Parameters,'PrintParameters',False)
    # Performance(model, Data, PrintParameters=PrintParameters)
    Performance_PCA(model, Data, VT,Dataspace.OutputScaler, PrintParameters=PrintParameters)

# ==============================================================================
# Functions used to asses performance of models

def Performance(model, Data, PrintParameters=False,fast_pred_var=True):
    df_list = Metrics(model,Data,fast_pred_var=fast_pred_var) # dict of pandas dfs with same keys as Data

    for i,df in enumerate(df_list):
        print("Output_{}".format(i))
        print(df)
        print()

        if PrintParameters:
            GPR.PrintParameters(model, output_ix=i)

def Performance_PCA(model,Data,VT,OutputScaler,PrintParameters=False, fast_pred_var=True):
    ''' Gets the averages of the metrics for the fully scaled version '''

    for key, val in Data.items():
        data_in,data_out = val
        data_out = data_out.detach().numpy()

        pred_mean = _pred(model,data_in,fast_pred_var=fast_pred_var)
        pred_mean_rescale = ML.DataRescale(pred_mean,*OutputScaler)
        # print(data_in[0].detach().numpy())
        # print(pred_mean_rescale[0])
        # print()
        data_out_rescale = ML.DataRescale(data_out,*OutputScaler)
        df_data_uncompress = ML.GetMetrics2(pred_mean_rescale.dot(VT),data_out_rescale.dot(VT))

        print('==============================================================')
        print('{}\n{}\n'.format(key,df_data_uncompress.mean()))

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


    NbOutput = (list(metrics.values())[0]).shape[0]
    column_names = (list(metrics.values())[0]).columns.values.tolist()
    index_names = (list(metrics.values())[0]).index.values.tolist()
    data_names = list(metrics.keys())

    df_list = []
    for i in range(NbOutput):
        dat = [df.iloc[i].tolist() for df in metrics.values()]
        a = pd.DataFrame(dat,columns=column_names,index=data_names)
        df_list.append(a)

    return df_list


def _pred(model,input,fast_pred_var=True):
    def _predfn(model,input):
        if hasattr(model,'models'):
            pred = model(*[input]*len(model.models))
            pred_mean = np.transpose([p.mean.numpy() for p in pred])
        else:
            pred_mean = model(input).mean.numpy()
        return pred_mean
    
    with torch.no_grad(), gpytorch.settings.debug(state=False), gpytorch.settings.fast_pred_var(fast_pred_var):
        pred_mean = _predfn(model,input)

    return pred_mean
