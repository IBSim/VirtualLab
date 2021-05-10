import numpy as np
import os
import matplotlib.pyplot as plt
from importlib import import_module, reload
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch
import h5py
import scipy

from Functions import DataScale, DataRescale
from CoilConfig_GPR import ExactGPmodel

def Single(VL, MLdict):
    Parameters = MLdict["Parameters"]

    ResDir = "{}/{}".format(VL.PROJECT_DIR, MLdict["Name"])

    # Methods we are considering
    Methods = ['Random','Grid','Adaptive','Adaptive_Uniform','Halton']

    # Get TestNb number of points from TestData
    DataFile = "{}/ML/Data.hdf5".format(VL.PROJECT_DIR)
    MLData = h5py.File(DataFile,'r')
    TestData = MLData["TestData/PU_2"][:Parameters.TestNb].astype('float32')

    DataDict = {}
    for ResName in os.listdir(ResDir):
        ResSubDir = "{}/{}".format(ResDir,ResName)

        if ResName.startswith('_'): continue
        if not os.path.isdir(ResSubDir): continue

        # import parameters used for ML
        sys.path.insert(0,ResSubDir)
        MLParameters = reload(import_module('Parameters'))
        sys.path.pop(0)

        # Check Method used in in Methods list
        Method = MLParameters.TrainData.split('/')[0]
        if Method.startswith('Grid'): Method = 'Grid'
        if Method == 'HIVE_Random': Method = 'Random'
        if Method not in Methods: continue

        if Method not in DataDict:
            DataDict[Method] = {'TrainNb':[],'TestMSE':[],'TrainMSE':[],'MaxPower':[]}

        # Get data used to train model
        TrainData = (np.load("{}/TrainData.npy".format(ResSubDir))).astype('float32')
        Train_x,Train_y = TrainData[:,:4],TrainData[:,4:]
        Test_x,Test_y = TestData[:,:4],TestData[:,4:]

        # Scale train and test data
        InputScaler = np.array([Train_x.min(axis=0),Train_x.max(axis=0) - Train_x.min(axis=0)])
        OutputScaler = np.array([Train_y.min(axis=0),Train_y.max(axis=0) - Train_y.min(axis=0)])

        Train_x_scale, Train_y_scale = DataScale(Train_x,*InputScaler), DataScale(Train_y,*OutputScaler)
        Test_x_scale,Test_y_scale = DataScale(Test_x,*InputScaler), DataScale(Test_y,*OutputScaler)
        Train_x_tf,Train_y_tf = torch.from_numpy(Train_x_scale),torch.from_numpy(Train_y_scale)
        Test_x_tf,Test_y_tf = torch.from_numpy(Test_x_scale),torch.from_numpy(Test_y_scale)

        # Define model
        PowerLH = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-5))
        Power = ExactGPmodel(Train_x_tf, Train_y_tf[:,0], PowerLH)
        state_dict_P = torch.load('{}/Power.pth'.format(ResSubDir))
        Power.load_state_dict(state_dict_P)
        PowerLH.eval(); Power.eval()

        with torch.no_grad(), gpytorch.settings.max_cholesky_size(1000), gpytorch.settings.debug(False):
            pred = Power(Test_x_tf)
            Test_MSE_P = np.mean((OutputScaler[1,0]*(pred.mean-Test_y_tf[:,0]).numpy())**2)
            pred = Power(Train_x_tf)
            Train_MSE_P = np.mean((OutputScaler[1,0]*(pred.mean-Train_y_tf[:,0]).numpy())**2)
            DataDict[Method]['TrainNb'].append(MLParameters.TrainNb)
            DataDict[Method]['TestMSE'].append(Test_MSE_P)
            DataDict[Method]['TrainMSE'].append(Train_MSE_P)

        # Check max Power accuracy
        MaxPowerERMES = '{}/MaxPower.rmed'.format(ResSubDir)
        if not os.path.isfile(MaxPowerERMES): continue

        ERMESres = h5py.File(MaxPowerERMES, 'r')
        attrs =  ERMESres["EM_Load"].attrs
        Scale = (1000/attrs['Current'])**2
        Watts = ERMESres["EM_Load/Watts"][:]*Scale
        JHNode =  ERMESres["EM_Load/JHNode"][:]*Scale
        ERMESres.close()

        Power = np.sum(Watts)
        DataDict[Method]['MaxPower'].append(Power)


    fig, axes = plt.subplots(nrows=2,ncols=1, sharex=True,figsize=(15,10))
    fig.suptitle('ML accuracy for different sampling methods and training dataset sizes',fontsize=18)
    for Name, dat in DataDict.items():
        # Get sort order and arrange loss values accordingly
        srt = np.argsort(dat['TrainNb'])
        for key in dat.keys():
            dat[key] = np.array(dat[key])[srt]

        axes[0].plot(dat['TrainNb'], dat['TestMSE'],label='{} ({} points)'.format(Name,Parameters.TestNb))
        axes[1].plot(dat['TrainNb'], dat['TrainMSE'],label=Name)

    axes[0].set_title('Test data',fontsize=14)
    axes[0].set_ylabel('MSE',fontsize=14)
    axes[0].legend(fontsize=14)
    axes[1].set_title('Train data',fontsize=14)
    axes[1].set_ylabel('MSE',fontsize=14)
    axes[1].set_xlabel('Number of data points used for training',fontsize=14)
    axes[1].legend(fontsize=14)
    plt.savefig("{}/MSE.png".format(ResDir))
    plt.close()

    nc = 1
    nr = int(np.ceil(len(DataDict)//nc))
    fig, axes = plt.subplots(nrows=nr,ncols=nc, sharex=True,sharey=True, figsize=(15,10))
    for i, (ax, (Name, dat)) in enumerate(zip(axes.flatten(),DataDict.items())):
        ax.plot(dat['TrainNb'],dat['MaxPower'])
        ax.set_title(Name, fontsize=14)
    fig.text(0.5, 0.04, 'Number of data points used for training', ha='center',fontsize=16)
    fig.text(0.04, 0.5, 'Power imparted to sample', va='center', rotation='vertical',fontsize=16)
    plt.savefig("{}/MaxPower.png".format(ResDir))
    plt.close()












        #
