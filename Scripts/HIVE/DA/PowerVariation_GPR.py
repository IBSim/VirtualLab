import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace as Namespace
import torch
import gpytorch
from PIL import Image

from VLFunctions import ReadData, ReadParameters
from Scripts.Common.ML import ML, Adaptive, GPR, NN
from Scripts.Common.tools import MEDtools

dtype = 'float64' # float64 is more accurate for optimisation purposes
torch_dtype = getattr(torch,dtype)
torch.set_default_dtype(torch_dtype)

def Variation(MeshFile,Data):
    ''' Calculate variation on CoilFace '''
    meshdata = MEDtools.MeshInfo(MeshFile)
    CoilFace = meshdata.GroupInfo('CoilFace')
    Connect = CoilFace.Connect
    Coor = meshdata.GetNodeXYZ(CoilFace.Nodes)
    meshdata.Close()

    _Ix = np.searchsorted(CoilFace.Nodes,Connect)
    elem_cd = Coor[_Ix]
    v1,v2 = elem_cd[:,1] - elem_cd[:,0], elem_cd[:,2] - elem_cd[:,0]
    cross = np.cross(v1,v2)
    area = 0.5*np.linalg.norm(cross,axis=1)

    Var = []
    for _data in Data:
        elem_cd[:,:,2] = _data[_Ix]/10**6
        v1,v2 = elem_cd[:,1] - elem_cd[:,0], elem_cd[:,2] - elem_cd[:,0]
        cross = np.cross(v1,v2)
        crossxy = cross[:,:2]/cross[:,2:3]
        crossmag = np.linalg.norm(crossxy,axis=1)
        _Var = (area*crossmag).sum()
        Var.append(_Var)
    return np.array(Var)


def Insight(VL,DADict):
    Parameters = DADict['Parameters']

    # ==========================================================================
    # Load power model & test data
    ModelDir_P = "{}/{}".format(VL.PROJECT_DIR,Parameters.PowerModelDir)
    likelihood_P, model_P, Dataspace_P, ParametersMod_P = GPR.LoadModel(ModelDir_P)

    DataFile_path = "{}/{}".format(VL.PROJECT_DIR, Parameters.PowerData[0])
    DataIn_P, DataOut_P = ML.GetDataML(DataFile_path, *Parameters.PowerData[1:])
    # DataOut_P = DataOut_P.flatten()
    ML.DataspaceAdd(Dataspace_P, Data=[DataIn_P,DataOut_P])

    # Load var model & test data
    ModelDir_V = "{}/{}".format(VL.PROJECT_DIR,Parameters.VarModelDir)
    likelihood_V, model_V, Dataspace_V, ParametersMod_V = GPR.LoadModel(ModelDir_V)

    DataFile_path = "{}/{}".format(VL.PROJECT_DIR, Parameters.VarData[0])
    DataIn_V, DataOut_V = ML.GetDataML(DataFile_path, *Parameters.VarData[1:])
    MeshFile = "{}/{}".format(VL.PROJECT_DIR,Parameters.MeshFile)
    DataOut_V = Variation(MeshFile,DataOut_V) # true var value from test data
    ML.DataspaceAdd(Dataspace_V, Data=[DataIn_V,DataOut_V])

    # set bounds for problem
    if hasattr(Parameters,'Bounds'):
        bounds = np.transpose(Parameters.Bounds)
        # scale bounds
        bounds = ML.DataScale(bounds,*Dataspace_P.InputScaler).T
    else:
        bounds = [[0,1]]*Dataspace_P.NbInput

    # ==========================================================================
    # Get min and max values for each
    print('Extrema')
    np.set_printoptions(precision=3)
    RangeDict = {}
    for name, mod, data in zip(['Power','Variation'],[model_P,model_V],[Dataspace_P,Dataspace_V]):
        val,cd = ML.GetExtrema(mod, 50, bounds)
        cd = ML.DataRescale(cd,*data.InputScaler)
        val = ML.DataRescale(val,*data.OutputScaler)
        RangeDict["Min_{}".format(name)] = val[0]
        RangeDict["Max_{}".format(name)] = val[1]
        print(name)
        print("Minima: Value:{:.2f}, Input: {}".format(val[0],cd[0]))
        print("Maxima: Value:{:.2f}, Input: {}\n".format(val[1],cd[1]))

    # ==========================================================================
    # Get minimum variation for different powers & plot
    MinVar_Constraint = getattr(Parameters,'MinVar_Constraint',{})
    if MinVar_Constraint:
        space = MinVar_Constraint.get('Space',50)
        rdlow = int(np.ceil(RangeDict['Min_Power'] / space)) * space
        rdhigh = int(np.ceil(RangeDict['Max_Power'] / space)) * space

        P = list(range(rdlow,rdhigh,space))
        Vmin,Vmax = [],[]
        for i in P:
            iscale = ML.DataScale(i,*Dataspace_P.OutputScaler)
            con = ML.FixedBound(model_P, iscale)
            # get minimum variation
            Opt_cd, Opt_val = ML.GetOptima(model_V, 100, bounds,
                                           find='min', constraints=con, maxiter=30)
            Opt_val = ML.DataRescale(Opt_val,*Dataspace_V.OutputScaler)
            Vmin.append(Opt_val[0])
            # get maximum variation
            Opt_cd, Opt_val = ML.GetOptima(model_V, 100, bounds,
                                           find='max', constraints=con, maxiter=30)
            Opt_val = ML.DataRescale(Opt_val,*Dataspace_V.OutputScaler)
            Vmax.append(Opt_val[0])

        if MinVar_Constraint.get('Plot',True):
            plt.figure()
            plt.xlabel('Power')
            plt.ylabel('Variation')
            plt.plot(P,Vmin,label='Min. Variation')
            plt.plot(P,Vmax,label='Max. Variation')
            plt.legend()
            plt.savefig("{}/Variation_constrained.png".format(DADict['CALC_DIR']))
            plt.close()

def PCA_check(VL,DADict):
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
        VT = ML.PCA(TrainOut,metric=getattr(Parameters,'Metric',{}))
        np.save("{}/VT.npy".format(DADict['CALC_DIR']),VT)

    data = []
    for i in range(VT.shape[0]):
        TrainOutCompress = TrainOut.dot(VT[:i+1].T)
        TestOutCompress = TestOut.dot(VT[:i+1].T)
        _list = []
        for orig, compress in zip([TrainOut,TestOut],[TrainOutCompress,TestOutCompress]):
            diff = np.abs(compress.dot(VT[:i+1]) - orig) # compare uncompressed and original
            absmaxix = np.unravel_index(np.argmax(np.abs(diff), axis=None), diff.shape)
            max_diff = diff[absmaxix]
            # _list.append(max_diff)
            _list.append((diff**2).mean())
        data.append(_list)
    data = np.array(data)
    p = data[1:,1]/data[:-1,1]
    ixs = np.where(p>0.99)[0]
    ix = ixs[0]

    plt.figure()
    plt.plot(data[:,0],label='Train')
    plt.plot(data[:,1],label='Test')
    plt.plot([ix,ix],[1e-1,1e14])
    plt.legend()
    plt.yscale('log')
    plt.ylim(bottom=1e-1)
    plt.show()


def Variation_Compare(VL,DADict):

    ''' Compare the performance of the surface model with variation model
        to calculate the value of variation.'''

    Parameters = DADict['Parameters']

    DataFile_path = "{}/{}".format(VL.PROJECT_DIR, Parameters.Data[0])
    DataIn, DataOut = ML.GetDataML(DataFile_path, *Parameters.Data[1:])

    MeshFile = "{}/{}".format(VL.PROJECT_DIR,Parameters.MeshFile)
    Var_true = Variation(MeshFile,DataOut) # true var value from test data

    # ==========================================================================
    # Load field model & test data
    ModelDir = "{}/{}".format(VL.PROJECT_DIR,Parameters.FieldModelDir)
    likelihood, model, Dataspace, ParametersMod = GPR.LoadModel(ModelDir)
    VT = np.load("{}/VT.npy".format(ModelDir))
    DataOut_compress = DataOut.dot(VT.T)
    ML.DataspaceAdd(Dataspace, Data=[DataIn,DataOut_compress])

    # Load field model & test data
    ModelDir = "{}/{}".format(VL.PROJECT_DIR,Parameters.VarModelDir)
    likelihood_V, model_V, Dataspace_V, ParametersMod_V = GPR.LoadModel(ModelDir)
    ML.DataspaceAdd(Dataspace_V, Data=[DataIn,Var_true])

    # ==========================================================================
    # Test data

    # field model
    with torch.no_grad():
        pred = model(*[Dataspace.DataIn_scale]*len(model.models))
    pred = np.transpose([p.mean.numpy() for p in pred])
    pred = ML.DataRescale(pred,*Dataspace.OutputScaler)
    pred = pred.dot(VT) # uncompress
    FieldPred = Variation(MeshFile,pred)
    # variation model
    with torch.no_grad():
        pred = model_V(Dataspace_V.DataIn_scale).mean.numpy()
    Var_pred = ML.DataRescale(pred,*Dataspace_V.OutputScaler) # calc variation


    metrics = ML.GetMetrics2(Var_pred,Var_true)
    print('Variation model (test)')
    print(metrics)
    print()

    metrics = ML.GetMetrics2(FieldPred,Var_true)
    print('Surface model (test)')
    print(metrics)
    print()

    # ==========================================================================
    # Train data

    # calculate var_true using varition model as this hasnt been compressed
    Var_true = ML.DataRescale(Dataspace_V.TrainOut_scale.detach().numpy(),
                             *Dataspace_V.OutputScaler)

    # field model
    with torch.no_grad():
        pred = model(*[Dataspace.TrainIn_scale]*len(model.models))
        pred = np.transpose([p.mean.numpy() for p in pred])
    pred = ML.DataRescale(pred,*Dataspace.OutputScaler)
    pred = pred.dot(VT) # uncompress
    FieldPred = Variation(MeshFile,pred) # calc variation
    # var model
    with torch.no_grad():
        pred = model_V(Dataspace_V.TrainIn_scale).mean.numpy()
    Var_pred = ML.DataRescale(pred,*Dataspace_V.OutputScaler)

    metrics = ML.GetMetrics2(Var_pred,Var_true)
    print('Variation model (test)')
    print(metrics)
    print()

    metrics = ML.GetMetrics2(FieldPred,Var_true)
    print('Surface model (test)')
    print(metrics)
    print()

# ==============================================================================
# Data collection function
def VarFromField(VL,DADict):
    Parameters = DADict['Parameters']
    DataFile = "{}/{}".format(VL.PROJECT_DIR,Parameters.DataFile) # full path

    DataGroup = getattr(Parameters,'DataGroup',None)
    field_data = ML.Readhdf(DataFile,Parameters.FieldName,group=DataGroup)

    MeshFile = "{}/{}".format(VL.PROJECT_DIR,Parameters.MeshFile)

    for data,newname in zip(field_data,Parameters.MaxName):
        Var = Variation(MeshFile,data)
        ML.Writehdf(DataFile,newname,Var,group=DataGroup)
