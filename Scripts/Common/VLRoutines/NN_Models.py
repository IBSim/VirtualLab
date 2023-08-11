
import os

import numpy as np
import torch

from Scripts.Common.ML import ML, NN

dtype = 'float64' # float64 is more accurate for optimisation purposes
torch_dtype = getattr(torch,dtype)
torch.set_default_dtype(torch_dtype)

def MLP_data(VL,DataDict):
    Parameters = DataDict['Parameters']

    NbTorchThread = getattr(Parameters,'NbTorchThread',None)
    if NbTorchThread: torch.set_num_threads(NbTorchThread)

    # set seed, if provided
    seed = getattr(Parameters,'Seed',None)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
    TrainInput,TrainOutput = np.array(Parameters.TrainData)
    TestInput,TestOutput = np.array(Parameters.ValidationData)
    if TrainInput.ndim==1: TrainInput = TrainInput.reshape((-1,1))
    if TrainOutput.ndim==1: TrainOutput = TrainOutput.reshape((-1,1))
    if TestInput.ndim==1: TestInput = TestInput.reshape((-1,1))
    if TestOutput.ndim==1: TestOutput = TestOutput.reshape((-1,1))

    # ==========================================================================
    # Get parameters and build model
    ModelParameters = getattr(Parameters,'ModelParameters',{})
    TrainingParameters = getattr(Parameters,'TrainingParameters',{})
    model, Dataspace = NN.BuildModel([TrainInput,TrainOutput],[TestInput,TestOutput],
                            DataDict['CALC_DIR'], ModelParameters=ModelParameters,
                            TrainingParameters=TrainingParameters)

    # data for which performance metrics will be evaluated
    Data = {'Train':[Dataspace.TrainIn_scale,Dataspace.TrainOut_scale],
            'Validation':[Dataspace.TestIn_scale,Dataspace.TestOut_scale]}
          
    # ==========================================================================
    # Get performance metric of model
    NN.Performance(model,Data)

def MLP_hdf5(VL,DataDict):
    Parameters = DataDict['Parameters']

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
                            DataDict['CALC_DIR'], ModelParameters=ModelParameters,
                            TrainingParameters=TrainingParameters)

    # ==========================================================================
    # Get performance metric of model
    Data = {'Train':[Dataspace.TrainIn_scale,Dataspace.TrainOut_scale],
            'Validation':[Dataspace.TestIn_scale,Dataspace.TestOut_scale]}

    NN.Performance(model,Data)

def MLP_PCA_hdf5(VL,DataDict):
    Parameters = DataDict['Parameters']

    NbTorchThread = getattr(Parameters,'NbTorchThread',None)
    if NbTorchThread: torch.set_num_threads(NbTorchThread)

    seed = getattr(Parameters,'Seed',None)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # ==========================================================================
    # Get Train & test data from file DataFile_path
    TrainIn, TrainOut = ML.VLGetDataML(VL,Parameters.TrainData)
    ScalePCA = ML.ScaleValues(TrainOut,scaling='centre')
    TrainOut_centre = ML.DataScale(TrainOut,*ScalePCA)

    TestIn, TestOut = ML.VLGetDataML(VL,Parameters.ValidationData)
    TestOut_centre = ML.DataScale(TestOut,*ScalePCA)

    np.save("{}/ScalePCA.npy".format(DataDict['CALC_DIR']),ScalePCA)

    # ==========================================================================
    # Compress data & save compression matrix in CALC_DIR
    VT_file = "{}/VT.npy".format(DataDict['CALC_DIR'])
    if os.path.isfile(VT_file) and not getattr(Parameters,'VT',True):
        VT = np.load(VT_file)
    else:
        metric = getattr(Parameters,'Metric',{})
        U,s,VT = ML.GetPC(TrainOut_centre,metric=metric,centre=False)# no need to centre as already done
        np.save(VT_file,VT)

    TrainOutCompress = TrainOut_centre.dot(VT.T)
    TestOutCompress = TestOut_centre.dot(VT.T)

    # ==========================================================================
    # Get parameters and build model

    ModelParameters = getattr(Parameters,'ModelParameters',{})
    TrainingParameters = getattr(Parameters,'TrainingParameters',{})

    model, Dataspace = NN.BuildModel([TrainIn,TrainOutCompress],[TestIn,TestOutCompress],
                            DataDict['CALC_DIR'], ModelParameters=ModelParameters,
                            TrainingParameters=TrainingParameters)

    # ==========================================================================
    # Get performance metric of model
    Data = {'Train':[Dataspace.TrainIn_scale,Dataspace.TrainOut_scale],
            'Validation':[Dataspace.TestIn_scale,Dataspace.TestOut_scale]}
    
    NN.Performance(model,Data)
    NN.Performance_PCA(model, Data, VT, Dataspace.OutputScaler, mean=True)
