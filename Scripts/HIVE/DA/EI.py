import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import minimize, differential_evolution, shgo, basinhopping
from types import SimpleNamespace as Namespace
import torch
import gpytorch
from skopt.sampler import Lhs
from importlib import import_module, reload

from CoilConfig_GPR import ExactGPmodel
from Functions import DataScale, DataRescale

def EIGF(y_Power,y_Var,NN,alpha):
    UQ_Score = alpha*y_Power.variance + (1-alpha)*y_Var.variance
    N_Score = alpha*(y_Power.mean - NN[:,0])**2 + (1-alpha)*(y_Var.mean - NN[:,1])**2
    Score = N_Score + UQ_Score
    Ix = np.argmax(Score,None)
    return Ix

def UQ(y_Power,y_Var,alpha):
    UQ_Score = alpha*y_Power.variance + (1-alpha)*y_Var.variance
    Ix = np.argmax(UQ_Score,None)
    return Ix



def Single(VL, MLdict):
    DA = MLdict['Parameters']

    ModelDir = DA.Path

    sys.path.insert(0,ModelDir)
    ModelParameters = reload(import_module('Parameters'))
    sys.path.pop(0)

    TrainData = np.load("{}/TrainData.npy".format(ModelDir))

    TrainData = TrainData.astype('float32')
    Train_x, Train_y = TrainData[:,:4], TrainData[:,4:]

    bounds = np.transpose([Train_x.min(axis=0),Train_x.max(axis=0)]).tolist()

    # Scale test & train input data to [0,1] (based on training data)
    InputScaler = np.array([Train_x.min(axis=0),Train_x.max(axis=0) - Train_x.min(axis=0)])
    # InputScaler = np.array([np.mean(Train_x,axis=0),np.std(Train_x,axis=0)])
    Train_x_scale = DataScale(Train_x,*InputScaler)
    # Scale test & train output data to [0,1] (based on training data)
    OutputScaler = np.array([Train_y.min(axis=0),Train_y.max(axis=0) - Train_y.min(axis=0)])
    # OutputScaler = np.array([np.mean(Train_y,axis=0),np.std(Train_y,axis=0)])
    Train_y_scale = DataScale(Train_y,*OutputScaler)

    Train_x_tf = torch.from_numpy(Train_x_scale)
    Train_y_tf = torch.from_numpy(Train_y_scale)

    Train_P_tf,Train_V_tf = Train_y_tf[:,0],Train_y_tf[:,1]

    sig = 0.00001*torch.ones(Train_x_tf.shape[0])
    PowerLH = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(sig)
    VarLH = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(sig)

    # If kernel is 'Matern' then nu must be specified in Parameters
    options = {}
    if hasattr(ModelParameters,'Nu'): options['nu']=ModelParameters.Nu

    Power = ExactGPmodel(Train_x_tf, Train_P_tf, PowerLH,
                    ModelParameters.Kernel,options)
    Variation = ExactGPmodel(Train_x_tf, Train_V_tf, VarLH,
                    ModelParameters.Kernel,options)

    # Power
    state_dict_P = torch.load('{}/Power.pth'.format(ModelDir))
    Power.load_state_dict(state_dict_P)
    PowerLH.eval(); Power.eval()

    # Variation
    state_dict_V = torch.load('{}/Variation.pth'.format(ModelDir))
    Variation.load_state_dict(state_dict_V)
    VarLH.eval(); Variation.eval()

    lhs = Lhs(criterion="maximin", iterations=10000)
    Candidates = lhs.generate(bounds, DA.NbCandidate)
    Candidates = np.array(Candidates)
    Cand_scale = DataScale(Candidates,*InputScaler)

    alpha = 1

    NN = []
    for c in Candidates:
        d = np.linalg.norm(Train_x - c,axis=1)
        ix = np.argmin(d)
        NN.append(Train_y_scale[ix,:])
    NN = np.array(NN)

    NewPoints = []
    for i in range(DA.NbNew):
        with torch.no_grad():
            x_scale = torch.tensor(Cand_scale, dtype=torch.float32)
            y_Power,y_Var = Power(x_scale),Variation(x_scale)

        if DA.Metric == 'EIGF':
            Ix = EIGF(y_Power,y_Var,NN,alpha)
            NN = np.delete(NN,Ix,axis=0)
        if DA.Metric == 'UQ':
            Ix = UQ(y_Power,y_Var,alpha)

        x = x_scale[Ix:Ix+1,:]
        New_P,New_V = y_Power.mean[Ix:Ix+1], y_Var.mean[Ix:Ix+1]

        # Update models for next point
        Power = Power.get_fantasy_model(x,New_P,noise=sig[:1])
        Variation = Variation.get_fantasy_model(x,New_V,noise=sig[:1])

        # Remove Ix from condidates
        Cand_scale = np.delete(Cand_scale,Ix,axis=0)

        NewPoints.append(Candidates[Ix])

    MLdict['NewPoints'] = NewPoints
















#
