import os
import sys
import shutil

import h5py
import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
import scipy.stats as stats
from importlib import import_module

import VLFunctions as VLF
from Scripts.Common.tools import MEDtools
from Scripts.Common.ML import ML, GPR

dtype = 'float64' # float64 is more accurate for optimisation purposes
torch_dtype = getattr(torch,dtype)
torch.set_default_dtype(torch_dtype)

def VerifyInputs(VL,DADict):
    Parameters = DADict['Parameters']

    # ==========================================================================
    # Load temperature field model & test data
    ModelDir_T = "{}/{}".format(VL.PROJECT_DIR,Parameters.TemperatureModelDir)
    likelihood_T, model_T, Dataspace_T, ParametersModT = GPR.LoadModel(ModelDir_T)

    DataFile_path = "{}/{}".format(VL.PROJECT_DIR, Parameters.TemperatureData[0])
    DataIn_T, DataOut_T = ML.GetDataML(DataFile_path, *Parameters.TemperatureData[1:])
    ML.DataspaceAdd(Dataspace_T, Data=[DataIn_T,DataOut_T])

    # ==========================================================================
    # Load VonMises field model & test data
    ModelDir_VM = "{}/{}".format(VL.PROJECT_DIR,Parameters.VMisModelDir)
    likelihood_VM, model_VM, Dataspace_VM, ParametersModVM = GPR.LoadModel(ModelDir_VM)

    DataFile_path = "{}/{}".format(VL.PROJECT_DIR, Parameters.VMisData[0])
    DataIn_VM, DataOut_VM = ML.GetDataML(DataFile_path, *Parameters.VMisData[1:])
    ML.DataspaceAdd(Dataspace_VM, Data=[DataIn_VM,DataOut_VM])

    # ==========================================================================
    # Make temperature predictions for train & test data using models
    with torch.no_grad():
        pred_T = model_T(*[Dataspace_T.DataIn_scale]*len(model_T.models))
        pred_T = np.transpose([p.mean.numpy() for p in pred_T])
    pred_T = ML.DataRescale(pred_T,*Dataspace_T.OutputScaler)

    with torch.no_grad():
        pred_VM = model_VM(*[Dataspace_VM.DataIn_scale]*len(model_VM.models))
        pred_VM = np.transpose([p.mean.numpy() for p in pred_VM])
    pred_VM = ML.DataRescale(pred_VM,*Dataspace_VM.OutputScaler)

    tdiff = np.abs(DataOut_T - pred_T)
    vmdiff = np.abs(DataOut_VM - pred_VM)/10**6

    print('Test data')
    print("Temperature: Max. diff. {:.3f} C, Avg. diff {:.3f} C".format(tdiff.max(),tdiff.mean()))
    print("VonMises: Max. diff. {:.3f} MPa, Avg. diff {:.3f} MPa".format(vmdiff.max(),vmdiff.mean()))
    print()

    # ==========================================================================
    # Make VonMises predictions for train & test data using models
    with torch.no_grad():
        pred_T = model_T(*[Dataspace_T.TrainIn_scale]*len(model_T.models))
        pred_T = np.transpose([p.mean.numpy() for p in pred_T])
    pred_T = ML.DataRescale(pred_T,*Dataspace_T.OutputScaler)

    with torch.no_grad():
        pred_VM = model_VM(*[Dataspace_T.TrainIn_scale]*len(model_VM.models))
        pred_VM = np.transpose([p.mean.numpy() for p in pred_VM])
    pred_VM = ML.DataRescale(pred_VM,*Dataspace_VM.OutputScaler)

    target_T = ML.DataRescale(Dataspace_T.TrainOut_scale.detach().numpy(),*Dataspace_T.OutputScaler)
    target_VM = ML.DataRescale(Dataspace_VM.TrainOut_scale.detach().numpy(),*Dataspace_VM.OutputScaler)
    tdiff = np.abs(target_T - pred_T)
    vmdiff = np.abs(target_VM - pred_VM)/10**6

    print('Train data')
    print("Temperature: Max. diff. {:.3f} C, Avg. diff {:.3f} C".format(tdiff.max(),tdiff.mean()))
    print("VonMises: Max. diff. {:.3f} MPa, Avg. diff {:.3f} MPa".format(vmdiff.max(),vmdiff.mean()))
    print()

    # Experiments = np.array(Parameters.Experiments)
    # Experiments_scale = torch.from_numpy(ML.DataScale(Experiments,*Dataspace_T.InputScaler))
    #
    # with torch.no_grad():
    #     pred_T = model_T(*[Experiments_scale]*len(model_T.models))
    #     pred_T = np.transpose([p.mean.numpy() for p in pred_T])
    # pred_T = ML.DataRescale(pred_T,*Dataspace_T.OutputScaler)
    #
    # with torch.no_grad():
    #     pred_VM = model_VM(*[Experiments_scale]*len(model_VM.models))
    #     pred_VM = np.transpose([p.mean.numpy() for p in pred_VM])
    # pred_VM = ML.DataRescale(pred_VM,*Dataspace_VM.OutputScaler)
    # print(pred_T.shape)
    # argmaxT = np.argmax(pred_T,axis=1)
    # argmaxVM = np.argmax(pred_VM,axis=1)
    # for i,(t,vm) in enumerate(zip(argmaxT,argmaxVM)):
    #     print(pred_T[i,t],t, pred_VM[i,vm]/10**6,vm)

def VerifyInputs_Surrogate(VL,DADict):
    Parameters = DADict['Parameters']

    # ==========================================================================
    # Load temperature field model & test data
    ModelDir_T = "{}/{}".format(VL.PROJECT_DIR,Parameters.TemperatureModelDir)
    likelihood_T, model_T, Dataspace_T, ParametersModT = GPR.LoadModel(ModelDir_T)
    VT_T = np.load("{}/VT.npy".format(ModelDir_T))

    DataFile_path = "{}/{}".format(VL.PROJECT_DIR, Parameters.TemperatureData[0])
    DataIn_T, DataOut_T = ML.GetDataML(DataFile_path, *Parameters.TemperatureData[1:])

    DataOut_T_compress = DataOut_T.dot(VT_T.T)
    ML.DataspaceAdd(Dataspace_T, Data=[DataIn_T,DataOut_T_compress])

    # ==========================================================================
    # Load VonMises field model & test data
    ModelDir_VM = "{}/{}".format(VL.PROJECT_DIR,Parameters.VMisModelDir)
    likelihood_VM, model_VM, Dataspace_VM, ParametersModVM = GPR.LoadModel(ModelDir_VM)
    VT_VM = np.load("{}/VT.npy".format(ModelDir_VM))

    DataFile_path = "{}/{}".format(VL.PROJECT_DIR, Parameters.VMisData[0])
    DataIn_VM, DataOut_VM = ML.GetDataML(DataFile_path, *Parameters.VMisData[1:])

    DataOut_VM_compress = DataOut_VM.dot(VT_VM.T)
    ML.DataspaceAdd(Dataspace_VM, Data=[DataIn_VM,DataOut_VM_compress])

    # ==========================================================================

    with torch.no_grad():
        pred_T = model_T(*[Dataspace_T.DataIn_scale]*len(model_T.models))
        pred_T = np.transpose([p.mean.numpy() for p in pred_T])
    pred_T = ML.DataRescale(pred_T,*Dataspace_T.OutputScaler)
    pred_T = pred_T.dot(VT_T)

    with torch.no_grad():
        pred_VM = model_VM(*[Dataspace_VM.DataIn_scale]*len(model_VM.models))
        pred_VM = np.transpose([p.mean.numpy() for p in pred_VM])
    pred_VM = ML.DataRescale(pred_VM,*Dataspace_VM.OutputScaler)
    pred_VM = pred_VM.dot(VT_VM)

    tdiff = np.abs(DataOut_T.max(axis=1) - pred_T.max(axis=1))
    vmdiff = np.abs(DataOut_VM.max(axis=1) - pred_VM.max(axis=1))/10**6

    print('Test data')
    print("Temperature: Max. diff. {:.3f} C, Avg. diff {:.3f} C".format(tdiff.max(),tdiff.mean()))
    print("VonMises: Max. diff. {:.3f} MPa, Avg. diff {:.3f} MPa".format(vmdiff.max(),vmdiff.mean()))
    print()

    with torch.no_grad():
        pred_T = model_T(*[Dataspace_T.TrainIn_scale]*len(model_T.models))
        pred_T = np.transpose([p.mean.numpy() for p in pred_T])
    pred_T = ML.DataRescale(pred_T,*Dataspace_T.OutputScaler)
    pred_T = pred_T.dot(VT_T)

    with torch.no_grad():
        pred_VM = model_VM(*[Dataspace_T.TrainIn_scale]*len(model_VM.models))
        pred_VM = np.transpose([p.mean.numpy() for p in pred_VM])
    pred_VM = ML.DataRescale(pred_VM,*Dataspace_VM.OutputScaler)
    pred_VM = pred_VM.dot(VT_VM)

    target_T = ML.DataRescale(Dataspace_T.TrainOut_scale.detach().numpy(),*Dataspace_T.OutputScaler)
    target_T = target_T.dot(VT_T)
    target_VM = ML.DataRescale(Dataspace_VM.TrainOut_scale.detach().numpy(),*Dataspace_VM.OutputScaler)
    target_VM = target_VM.dot(VT_VM)
    tdiff = np.abs(target_T.max(axis=1) - pred_T.max(axis=1))
    vmdiff = np.abs(target_VM.max(axis=1) - pred_VM.max(axis=1))/10**6
    print('Train data')
    print("Temperature: Max. diff. {:.3f} C, Avg. diff {:.3f} C".format(tdiff.max(),tdiff.mean()))
    print("Von Mises: Max. diff. {:.3f} MPa, Avg. diff {:.3f} MPa".format(vmdiff.max(),vmdiff.mean()))
    print()

    return



    # with torch.no_grad():
    #     pred_T = model_T(*[Dataspace_T.DataIn_scale]*len(model_T.models))
    #     pred_T_test = np.transpose([p.mean.numpy() for p in pred_T])
    # pred_T_test = ML.DataRescale(pred_T_test,*Dataspace_T.OutputScaler)
    #
    # with torch.no_grad():
    #     pred_T = model_T(*[Dataspace_T.TrainIn_scale]*len(model_T.models))
    #     pred_T_train = np.transpose([p.mean.numpy() for p in pred_T])
    # pred_T_train = ML.DataRescale(pred_T_train,*Dataspace_T.OutputScaler)
    #
    #
    # for i in range(1,1+VT_T.shape[0]):
    #     pred_test = pred_T_test[:,:i].dot(VT_T[:i,:])
    #     pred_train = pred_T_train[:,:i].dot(VT_T[:i,:])
    #     df_test = ML.GetMetrics2(pred_test,DataOut_T)
    #     df_train = ML.GetMetrics2(pred_train,target_T)
    #     print(i)
    #     print(df_train.mean())
    #     print(df_test.mean())
    #     print()

    # Experiments = np.array(Parameters.Experiments)
    # Experiments_scale = torch.from_numpy(ML.DataScale(Experiments,*Dataspace_T.InputScaler))
    #
    # with torch.no_grad():
    #     pred_T = model_T(*[Experiments_scale]*len(model_T.models))
    #     pred_T = np.transpose([p.mean.numpy() for p in pred_T])
    # pred_T = ML.DataRescale(pred_T,*Dataspace_T.OutputScaler)
    # pred_T = pred_T.dot(VT_T)
    #
    # with torch.no_grad():
    #     pred_VM = model_VM(*[Experiments_scale]*len(model_VM.models))
    #     pred_VM = np.transpose([p.mean.numpy() for p in pred_VM])
    # pred_VM = ML.DataRescale(pred_VM,*Dataspace_VM.OutputScaler)
    # pred_VM = pred_VM.dot(VT_VM)
    #
    # argmaxT = np.argmax(pred_T,axis=1)
    # argmaxVM = np.argmax(pred_VM,axis=1)
    # for i,(t,vm) in enumerate(zip(argmaxT,argmaxVM)):
    #     print(pred_T[i,t],t, pred_VM[i,vm]/10**6,vm)

def Field_v_Max(VL,DADict):
    Parameters = DADict['Parameters']

    # ==========================================================================
    # Load field model & test data
    ModelDir_field = "{}/{}".format(VL.PROJECT_DIR,Parameters.ModelDir_Field)
    likelihood_field, model_field, Dataspace_field, ParametersMod_field = GPR.LoadModel(ModelDir_field)
    VT = np.load("{}/VT.npy".format(ModelDir_field))

    DataFile_path = "{}/{}".format(VL.PROJECT_DIR, Parameters.DataField[0])
    DataIn_field, DataOut_field = ML.GetDataML(DataFile_path, *Parameters.DataField[1:])
    DataOut_field_compress = DataOut_field.dot(VT.T)
    ML.DataspaceAdd(Dataspace_field, Data=[DataIn_field,DataOut_field_compress])

    # ==========================================================================
    # Load max. model & test data
    ModelDir_max = "{}/{}".format(VL.PROJECT_DIR,Parameters.ModelDir_Max)
    likelihood_max, model_max, Dataspace_max, ParametersMod_max = GPR.LoadModel(ModelDir_max)

    DataFile_path = "{}/{}".format(VL.PROJECT_DIR, Parameters.DataMax[0])
    DataIn_max, DataOut_max = ML.GetDataML(DataFile_path, *Parameters.DataMax[1:])
    ML.DataspaceAdd(Dataspace_max, Data=[DataIn_max,DataOut_max])

    print((DataIn_max==DataIn_field).all()) # data sets are the same

    # ==========================================================================
    # Make predictions for train & test data using max. model
    with torch.no_grad():
        pred_max_test = model_max(*[Dataspace_max.DataIn_scale]*len(model_max.models))
        pred_max_test = np.transpose([p.mean.numpy() for p in pred_max_test])
        pred_max_train = model_max(*[Dataspace_max.TrainIn_scale]*len(model_max.models))
        pred_max_train = np.transpose([p.mean.numpy() for p in pred_max_train])

    pred_max_test = ML.DataRescale(pred_max_test,*Dataspace_max.OutputScaler)
    pred_max_train = ML.DataRescale(pred_max_train,*Dataspace_max.OutputScaler)

    tdiff_test = np.abs(DataOut_max - pred_max_test)
    _target = ML.DataRescale(Dataspace_max.TrainOut_scale.detach().numpy(),*Dataspace_max.OutputScaler)
    tdiff_train = np.abs(_target - pred_max_train)

    print('Max. model')
    print("Test data: Max. diff. {:.3f}, Avg. diff {:.3f}".format(tdiff_test.max(),tdiff_test.mean()))
    print("Train data: Max. diff. {:.3f}, Avg. diff {:.3f}".format(tdiff_train.max(),tdiff_train.mean()))
    print()

    # ==========================================================================
    # Make predictions for train & test data using field. model
    with torch.no_grad():
        pred_field_test = model_field(*[Dataspace_field.DataIn_scale]*len(model_field.models))
        pred_field_test = np.transpose([p.mean.numpy() for p in pred_field_test])
        pred_field_train = model_field(*[Dataspace_field.TrainIn_scale]*len(model_field.models))
        pred_field_train = np.transpose([p.mean.numpy() for p in pred_field_train])

    pred_field_test = ML.DataRescale(pred_field_test,*Dataspace_field.OutputScaler)
    pred_field_test = pred_field_test.dot(VT)
    pred_field_train = ML.DataRescale(pred_field_train,*Dataspace_field.OutputScaler)
    pred_field_train = pred_field_train.dot(VT)

    tdiff_test = np.abs(DataOut_field.max(axis=1) - pred_field_test.max(axis=1))
    _target = ML.DataRescale(Dataspace_field.TrainOut_scale.detach().numpy(),*Dataspace_field.OutputScaler)
    _target = _target.dot(VT)
    tdiff_train = np.abs(_target.max(axis=1) - pred_field_train.max(axis=1))

    print('Field model')
    print("Test data: Max. diff. {:.3f}, Avg. diff {:.3f}".format(tdiff_test.max(),tdiff_test.mean()))
    print("Train data: Max. diff. {:.3f}, Avg. diff {:.3f}".format(tdiff_train.max(),tdiff_train.mean()))
    print()

    # print((Dataspace_max.DataIn_scale==Dataspace_field.DataIn_scale).all())
    # print((Dataspace_max.TrainIn_scale==Dataspace_field.TrainIn_scale).all())
    # print((DataOut_field.max(axis=1) == DataOut_max.flatten()).all())


# ==============================================================================
# Data collection function
def MaxFromField(VL,DADict):
    Parameters = DADict['Parameters']
    DataFile = "{}/{}".format(VL.PROJECT_DIR,Parameters.DataFile) # full path

    DataGroup = getattr(Parameters,'DataGroup',None)
    field_data = ML.Readhdf(DataFile,Parameters.FieldName,group=DataGroup)

    for data,newname in zip(field_data,Parameters.MaxName):
        max_data = data.max(axis=1)
        max_data = max_data.reshape((max_data.size,1)) #ensure 2 dim

        ML.Writehdf(DataFile,newname,max_data,group=DataGroup)
