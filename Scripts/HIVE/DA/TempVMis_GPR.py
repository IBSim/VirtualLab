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
from Scripts.Common.ML import ML
from Scripts.Common.Optimisation import slsqp_multi, GA, GA_Parallel

dtype = 'float64' # float64 is more accurate for optimisation purposes
torch_dtype = getattr(torch,dtype)
torch.set_default_dtype(torch_dtype)

def Test(VL,DADict):
    Parameters = DADict['Parameters']

    # ==========================================================================
    # Load temperature field model
    ModelDir_T = "{}/{}".format(VL.PROJECT_DIR,Parameters.TemperatureModelDir)
    likelihood_T, model_T, Dataspace_T, ParametersModT = ML.Load_GPR(ModelDir_T)
    VT_T = np.load("{}/VT.npy".format(ModelDir_T))

    DataFile_path = "{}/{}".format(VL.PROJECT_DIR, Parameters.DataFile)
    DataIn_T, DataOut_T = ML.GetMLdata(DataFile_path, Parameters.DataT,
                                   Parameters.InputT, Parameters.OutputT,
                                   getattr(Parameters,'TestNb',-1))

    DataOut_T_compress = DataOut_T.dot(VT_T.T)
    ML.DataspaceAdd(Dataspace_T, Data=[DataIn_T,DataOut_T_compress])

    # ==========================================================================
    # Load VonMises field model
    ModelDir_VM = "{}/{}".format(VL.PROJECT_DIR,Parameters.VMisModelDir)
    likelihood_VM, model_VM, Dataspace_VM, ParametersModVM = ML.Load_GPR(ModelDir_VM)
    VT_VM = np.load("{}/VT.npy".format(ModelDir_VM))

    DataIn_VM, DataOut_VM = ML.GetMLdata(DataFile_path, Parameters.DataVMis,
                                   Parameters.InputVMis, Parameters.OutputVMis,
                                   getattr(Parameters,'TestNb',-1))

    DataOut_VM_compress = DataOut_VM.dot(VT_VM.T)
    ML.DataspaceAdd(Dataspace_VM, Data=[DataIn_VM,DataOut_VM_compress])



    # ix=2
    # with torch.no_grad():
    #     pred_T = model_T(*[Dataspace_T.DataIn_scale]*len(model_T.models))
    #     pred_T = np.transpose([p.mean.numpy() for p in pred_T])
    # pred_T = ML.DataRescale(pred_T,*Dataspace_T.OutputScaler)
    # pred_T = pred_T.dot(VT_T)
    # for i in range(20):
    #     print(pred_T[ix,i], DataOut_T[ix,i])
    #
    # print()
    # with torch.no_grad():
    #     pred_VM = model_VM(*[Dataspace_VM.DataIn_scale]*len(model_VM.models))
    #     pred_VM = np.transpose([p.mean.numpy() for p in pred_VM])
    # pred_VM = ML.DataRescale(pred_VM,*Dataspace_VM.OutputScaler)
    # pred_VM = pred_VM.dot(VT_VM)
    # for i in range(20):
    #     print(pred_VM[ix,i], DataOut_VM[ix,i],(pred_VM[ix,i]/DataOut_VM[ix,i]) -1)

def VerifyInputs(VL,DADict):
    Parameters = DADict['Parameters']

    # ==========================================================================
    # Load temperature field model
    ModelDir_T = "{}/{}".format(VL.PROJECT_DIR,Parameters.TemperatureModelDir)
    likelihood_T, model_T, Dataspace_T, ParametersModT = ML.Load_GPR(ModelDir_T)

    DataFile_path = "{}/{}".format(VL.PROJECT_DIR, Parameters.DataFile)
    DataIn_T, DataOut_T = ML.GetMLdata(DataFile_path, Parameters.DataT,
                                   Parameters.InputT, Parameters.OutputT,
                                   getattr(Parameters,'TestNb',-1))

    ML.DataspaceAdd(Dataspace_T, Data=[DataIn_T,DataOut_T])

    # ==========================================================================
    # Load VonMises field model
    ModelDir_VM = "{}/{}".format(VL.PROJECT_DIR,Parameters.VMisModelDir)
    likelihood_VM, model_VM, Dataspace_VM, ParametersModVM = ML.Load_GPR(ModelDir_VM)

    DataIn_VM, DataOut_VM = ML.GetMLdata(DataFile_path, Parameters.DataVMis,
                                   Parameters.InputVMis, Parameters.OutputVMis,
                                   getattr(Parameters,'TestNb',-1))

    ML.DataspaceAdd(Dataspace_VM, Data=[DataIn_VM,DataOut_VM])


    # ==========================================================================

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

    return

    Experiments = np.array(Parameters.Experiments)
    Experiments_scale = torch.from_numpy(ML.DataScale(Experiments,*Dataspace_T.InputScaler))

    with torch.no_grad():
        pred_T = model_T(*[Experiments_scale]*len(model_T.models))
        pred_T = np.transpose([p.mean.numpy() for p in pred_T])
    pred_T = ML.DataRescale(pred_T,*Dataspace_T.OutputScaler)

    with torch.no_grad():
        pred_VM = model_VM(*[Experiments_scale]*len(model_VM.models))
        pred_VM = np.transpose([p.mean.numpy() for p in pred_VM])
    pred_VM = ML.DataRescale(pred_VM,*Dataspace_VM.OutputScaler)
    print(pred_T.shape)
    argmaxT = np.argmax(pred_T,axis=1)
    argmaxVM = np.argmax(pred_VM,axis=1)
    for i,(t,vm) in enumerate(zip(argmaxT,argmaxVM)):
        print(pred_T[i,t],t, pred_VM[i,vm]/10**6,vm)

def VerifyInputs_Surrogate(VL,DADict):
    Parameters = DADict['Parameters']

    # ==========================================================================
    # Load temperature field model
    ModelDir_T = "{}/{}".format(VL.PROJECT_DIR,Parameters.TemperatureModelDir)
    likelihood_T, model_T, Dataspace_T, ParametersModT = ML.Load_GPR(ModelDir_T)
    VT_T = np.load("{}/VT.npy".format(ModelDir_T))

    DataFile_path = "{}/{}".format(VL.PROJECT_DIR, Parameters.DataFile)
    DataIn_T, DataOut_T = ML.GetMLdata(DataFile_path, Parameters.DataT,
                                   Parameters.InputT, Parameters.OutputT,
                                   getattr(Parameters,'TestNb',-1))

    DataOut_T_compress = DataOut_T.dot(VT_T.T)
    ML.DataspaceAdd(Dataspace_T, Data=[DataIn_T,DataOut_T_compress])

    # ==========================================================================
    # Load VonMises field model
    ModelDir_VM = "{}/{}".format(VL.PROJECT_DIR,Parameters.VMisModelDir)
    likelihood_VM, model_VM, Dataspace_VM, ParametersModVM = ML.Load_GPR(ModelDir_VM)
    VT_VM = np.load("{}/VT.npy".format(ModelDir_VM))

    DataIn_VM, DataOut_VM = ML.GetMLdata(DataFile_path, Parameters.DataVMis,
                                   Parameters.InputVMis, Parameters.OutputVMis,
                                   getattr(Parameters,'TestNb',-1))

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


    Experiments = np.array(Parameters.Experiments)
    Experiments_scale = torch.from_numpy(ML.DataScale(Experiments,*Dataspace_T.InputScaler))

    with torch.no_grad():
        pred_T = model_T(*[Experiments_scale]*len(model_T.models))
        pred_T = np.transpose([p.mean.numpy() for p in pred_T])
    pred_T = ML.DataRescale(pred_T,*Dataspace_T.OutputScaler)
    pred_T = pred_T.dot(VT_T)

    with torch.no_grad():
        pred_VM = model_VM(*[Experiments_scale]*len(model_VM.models))
        pred_VM = np.transpose([p.mean.numpy() for p in pred_VM])
    pred_VM = ML.DataRescale(pred_VM,*Dataspace_VM.OutputScaler)
    pred_VM = pred_VM.dot(VT_VM)

    argmaxT = np.argmax(pred_T,axis=1)
    argmaxVM = np.argmax(pred_VM,axis=1)
    for i,(t,vm) in enumerate(zip(argmaxT,argmaxVM)):
        print(pred_T[i,t],t, pred_VM[i,vm]/10**6,vm)


# ==============================================================================
# Functions for gathering necessary data and writing to file
def CompileData(VL,DADict):
    Parameters = DADict["Parameters"]

    # ==========================================================================
    # Get list of all the results directories which will be searched through
    CmpData = Parameters.CompileData
    if type(CmpData)==str:CmpData = [CmpData]
    ResDirs = ["{}/{}".format(VL.PROJECT_DIR,resname) for resname in CmpData]

    # ==========================================================================
    # Specify the function used to gather the necessary data & any arguments required
    args= []
    if Parameters.OutputFn.lower()=="fieldtemperatures":
        OutputFn = _FieldTemperatures
        args = [Parameters.InputVariables, Parameters.ResFileName]
    elif Parameters.OutputFn.lower()=="fieldvmis":
        OutputFn = _FieldVMis
        args = [Parameters.InputVariables, Parameters.ResFileName]

    # ==========================================================================
    # Apply OutputFn to all sub dirs in ResDirs
    InData, OutData = ML.CompileData(ResDirs,OutputFn,args=args)

    # ==========================================================================
    # Write the input and output data to DataFile_path
    DataFile_path = "{}/{}".format(VL.PROJECT_DIR,Parameters.DataFile)

    ML.WriteMLdata(DataFile_path, CmpData, Parameters.InputArray, InData,
                    attrs=getattr(Parameters,'InputAttributes',{}))
    ML.WriteMLdata(DataFile_path, CmpData, Parameters.OutputArray, OutData,
                    attrs=getattr(Parameters,'OutputAttributes',{}))


# ==============================================================================
# Data collection functions
def _FieldTemperatures(ResDir, InputVariables, ResFileName, ResName='Temperature'):
    ''' Get temperature values at all nodes'''


    paramfile = "{}/Parameters.py".format(ResDir)
    Parameters = VLF.ReadParameters(paramfile)
    In = ML.GetInputs(Parameters,InputVariables)

    # Get temperature values from results
    ResFilePath = "{}/{}".format(ResDir,ResFileName)
    Out = MEDtools.FieldResult(ResFilePath,ResName)

    return In, Out

def _FieldVMis(ResDir, InputVariables, ResFileName, ResName='Stress'):
    ''' Get temperature values at all nodes'''

    paramfile = "{}/Parameters.py".format(ResDir)
    Parameters = VLF.ReadParameters(paramfile)
    In = ML.GetInputs(Parameters,InputVariables)

    # Get temperature values from results
    ResFilePath = "{}/{}".format(ResDir,ResFileName)
    Stress = MEDtools.ElementResult(ResFilePath,ResName)
    Stress = Stress.reshape((int(Stress.size/6),6))

    VMis = (((Stress[:,0] - Stress[:,1])**2 + (Stress[:,1] - Stress[:,2])**2 + \
              (Stress[:,2] - Stress[:,0])**2 + 6*(Stress[:,3:]**2).sum(axis=1)  )/2)**0.5

    mesh = MEDtools.MeshInfo(ResFilePath)
    cnct = mesh.ConnectByType('Volume')

    # average out to node
    sumvmis,sumcount = np.zeros(mesh.NbNodes),np.zeros(mesh.NbNodes)
    for i,vm in zip(cnct,VMis):
        sumvmis[i-1]+=vm
        sumcount[i-1]+=1
    VMis_nd = sumvmis/sumcount

    return In, VMis_nd
