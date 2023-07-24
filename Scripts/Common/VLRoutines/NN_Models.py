
import os
import sys
import shutil
import pandas as pd

import numpy as np
import torch
import gpytorch

from Scripts.Common.tools import MEDtools
from Scripts.Common.ML import ML, NN, Adaptive

dtype = 'float64' # float64 is more accurate for optimisation purposes
torch_dtype = getattr(torch,dtype)
torch.set_default_dtype(torch_dtype)

def MLP_hdf5(VL,DADict):
    Parameters = DADict['Parameters']

    NbTorchThread = getattr(Parameters,'NbTorchThread',None)
    if NbTorchThread: torch.set_num_threads(NbTorchThread)

    # set seed, if provided
    seed = getattr(Parameters,'Seed',None)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # ==========================================================================
    # Get Train & test data from file DataFile_path
    TrainIn, TrainOut = ML.VLGetDataML(VL,Parameters.TrainData)
    TestIn, TestOut = ML.VLGetDataML(VL,Parameters.ValidationData)

    # ==========================================================================
    # Get parameters and build model

    ModelParameters = getattr(Parameters,'ModelParameters',{})
    TrainingParameters = getattr(Parameters,'TrainingParameters',{})
    model, Dataspace = NN.BuildModel([TrainIn,TrainOut],[TestIn,TestOut],
                            DADict['CALC_DIR'], ModelParameters=ModelParameters,
                            TrainingParameters=TrainingParameters)

    # ==========================================================================
    # Get performance metric of model
    Data = {'Train':[Dataspace.TrainIn_scale,Dataspace.TrainOut_scale],
            'Test':[Dataspace.TestIn_scale,Dataspace.TestOut_scale]}

    NN.Performance(model,Data)

def MLP_PCA_hdf5(VL,DADict):
    Parameters = DADict['Parameters']

    NbTorchThread = getattr(Parameters,'NbTorchThread',None)
    if NbTorchThread: torch.set_num_threads(NbTorchThread)

    seed = getattr(Parameters,'Seed',None)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

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

    # ==========================================================================
    # Get parameters and build model

    ModelParameters = getattr(Parameters,'ModelParameters',{})
    TrainingParameters = getattr(Parameters,'TrainingParameters',{})

    model, Dataspace = NN.BuildModel([TrainIn,TrainOutCompress],[TestIn,TestOutCompress],
                            DADict['CALC_DIR'], ModelParameters=ModelParameters,
                            TrainingParameters=TrainingParameters)

    # ==========================================================================
    # Get performance metric of model
    Data = {'Train':[Dataspace.TrainIn_scale,Dataspace.TrainOut_scale],
            'Test':[Dataspace.TestIn_scale,Dataspace.TestOut_scale]}

    NN.Performance(model,Data)
    NN.Performance_PCA(model, Data, VT, Dataspace.OutputScaler, mean=True)
