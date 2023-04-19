import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace as Namespace
import torch
from PIL import Image

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data

import VLFunctions as VLF
from Scripts.Common.ML import ML, NN
from Scripts.Common.tools import MEDtools

dtype = 'float64' # float64 is more accurate for optimisation purposes
torch_dtype = getattr(torch,dtype)
torch.set_default_dtype(torch_dtype)

def BuildModel(VL,DADict):
    Parameters = DADict['Parameters']

    seed = getattr(Parameters,'Seed',None)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # ==========================================================================
    # Get Train & test data from file DataFile_path

    DataFile_path = "{}/{}".format(VL.PROJECT_DIR, Parameters.TrainData[0])
    TrainIn, TrainOut = ML.GetDataML(DataFile_path, *Parameters.TrainData[1:])
    TrainOut = TrainOut.reshape((int(TrainOut.size/2),2),order='F')

    DataFile_path = "{}/{}".format(VL.PROJECT_DIR, Parameters.TestData[0])
    TestIn, TestOut = ML.GetDataML(DataFile_path, *Parameters.TestData[1:])
    TestOut = TestOut.reshape((int(TestOut.size/2),2),order='F')

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



def PCA_unscaled(VL,DADict):

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

    # Create dataspace to conveniently keep data together
    Dataspace = ML.DataspaceTrain([TrainIn,TrainOutCompress],Test=[TestIn,TestOutCompress])
    Dataspace.TrainOut_scale = torch.from_numpy(TrainOutCompress)
    Dataspace.TestOut_scale = torch.from_numpy(TestOutCompress)

    # ==========================================================================
    # get model
    # Add input and output to architecture
    _architecture = ModelParameters.get('Architecture')
    if _architecture is None: sys.exit('Must have architecture')

    ModelParameters['Architecture'] = NN.AddIO(ModelParameters['Architecture'],Dataspace)

    model = NN.NN_FC(**ModelParameters)

    NbWeights = NN.GetNbWeights(model)
    ML.ModelSummary(Dataspace.NbInput,Dataspace.NbOutput,Dataspace.NbTrain,
                    NbWeights=NbWeights)

    # Train model
    Convergence = NN.TrainModel(model, Dataspace, **TrainingParameters)
    model.eval()

    NN.SaveModel(DADict['CALC_DIR'],model,TrainIn,TrainOutCompress,Convergence)

    # ==========================================================================
    # Get performance metric of model
    Data = {'Train':[Dataspace.TrainIn_scale,Dataspace.TrainOut_scale],
            'Test':[Dataspace.TestIn_scale,Dataspace.TestOut_scale]}

    NN.Performance(model,Data)
    NN.Performance_PCA(model, Data, VT, Dataspace.OutputScaler, mean=True)
