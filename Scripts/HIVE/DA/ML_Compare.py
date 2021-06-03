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
import pickle

from Functions import DataScale, DataRescale
from CoilConfig_GPR import ExactGPmodel, MSE

def Single(VL, MLdict):
    Parameters = MLdict["Parameters"]

    ResDir = "{}/{}".format(VL.PROJECT_DIR, MLdict["Name"])

    # Methods we are considering
    Methods = ['Random','Grid','Adaptive','Adaptive_Uniform','Halton',
                'Coupled_RBF','Coupled_Matern_1.5','Coupled_Matern_2.5']

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
            DataDict[Method] = {'TrainNb':[],'TestMSE_Power':[],'TrainMSE_Power':[],
                            'TestMSE_Variation':[],'TrainMSE_Variation':[],
                            'MaxPower_pred':[],'MaxPower_act':[]}

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

        with open("{}/Data.pkl".format(ResSubDir),'rb') as f:
            Data = pickle.load(f)

        DataDict[Method]['TrainNb'].append(MLParameters.TrainNb)

        # Likelihood
        if getattr(MLParameters,'Noise',True):
            PowerLH = gpytorch.likelihoods.GaussianLikelihood()
            VarLH = gpytorch.likelihoods.GaussianLikelihood()
        else:
            sig = 0.00001*torch.ones(Train_x_tf.shape[0])
            PowerLH = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(sig)
            VarLH = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(sig)

        options = {}
        if hasattr(MLParameters,'Nu'): options['nu']=MLParameters.Nu

        # Power model
        Power = ExactGPmodel(Train_x_tf, Train_y_tf[:,0], PowerLH,
                        MLParameters.Kernel,options)

        state_dict_P = torch.load('{}/Power.pth'.format(ResSubDir))
        Power.load_state_dict(state_dict_P)
        PowerLH.eval(); Power.eval()

        with torch.no_grad(), gpytorch.settings.max_cholesky_size(1000), gpytorch.settings.debug(False):
            Test_MSE_P = MSE(Power(Test_x_tf).mean.numpy(), Test_y_scale[:,0])*OutputScaler[1,0]**2
            Train_MSE_P = MSE(Power(Train_x_tf).mean.numpy(), Train_y_scale[:,0])*OutputScaler[1,0]**2
        print(Method,MLParameters.TrainNb,Test_MSE_P)
        DataDict[Method]['TestMSE_Power'].append(Test_MSE_P)
        DataDict[Method]['TrainMSE_Power'].append(Train_MSE_P)

        # Variation model
        Variation = ExactGPmodel(Train_x_tf, Train_y_tf[:,1], VarLH,
                        MLParameters.Kernel,options)

        state_dict_V = torch.load('{}/Variation.pth'.format(ResSubDir))
        Variation.load_state_dict(state_dict_V)
        VarLH.eval(); Variation.eval()

        with torch.no_grad(), gpytorch.settings.max_cholesky_size(1000), gpytorch.settings.debug(False):
            Test_MSE_V = MSE(Variation(Test_x_tf).mean.numpy(), Test_y_scale[:,1])*OutputScaler[1,1]**2
            Train_MSE_V = MSE(Variation(Train_x_tf).mean.numpy(), Train_y_scale[:,1])*OutputScaler[1,1]**2

        DataDict[Method]['TestMSE_Variation'].append(Test_MSE_V)
        DataDict[Method]['TrainMSE_Variation'].append(Train_MSE_V)


        Power_y = Data['MaxPower']['y']

        if 'target' in Data['MaxPower']:
            Power_target = Data['MaxPower']['target']
        elif os.path.isfile('{}/MaxPower.rmed'.format(ResSubDir)):

            ERMESres = h5py.File('{}/MaxPower.rmed'.format(ResSubDir), 'r')
            attrs =  ERMESres["EM_Load"].attrs
            Scale = (1000/attrs['Current'])**2
            Watts = ERMESres["EM_Load/Watts"][:]*Scale
            JHNode =  ERMESres["EM_Load/JHNode"][:]*Scale
            ERMESres.close()
            Power_target = np.sum(Watts)
        else:
            Power_target = 0

        DataDict[Method]['MaxPower_pred'].append(Power_y)
        DataDict[Method]['MaxPower_act'].append(Power_target)


    for res in ['Power','Variation']:
        fig, axes = plt.subplots(nrows=2,ncols=1, sharex=True,figsize=(15,10))
        fig.suptitle('MSE for {} based on different sampling methods\n and training dataset sizes'.format(res),fontsize=18)
        for Name, dat in DataDict.items():
            # Get sort order and arrange loss values accordingly
            srt = np.argsort(dat['TrainNb'])
            for key in dat.keys():
                print(dat[key])
                dat[key] = np.array(dat[key])[srt]

            axes[0].plot(dat['TrainNb'], dat['TestMSE_{}'.format(res)],label='{} ({} points)'.format(Name,Parameters.TestNb))
            axes[1].plot(dat['TrainNb'], dat['TrainMSE_{}'.format(res)],label=Name)

        axes[0].set_title('Test data',fontsize=14)
        axes[0].set_ylabel('MSE',fontsize=14)
        axes[0].legend(fontsize=14)
        axes[1].set_title('Train data',fontsize=14)
        axes[1].set_ylabel('MSE',fontsize=14)
        axes[1].set_xlabel('Number of data points used for training',fontsize=14)
        axes[1].legend(fontsize=14)
        plt.savefig("{}/MSE_{}.png".format(ResDir,res))
        plt.close()

    nc = 1
    nr = int(np.ceil(len(DataDict)/nc))
    fig, axes = plt.subplots(nrows=nr,ncols=nc, sharex=True,sharey=True, figsize=(15,10))
    if nr==1:axes=np.array([axes])
    for i, (ax, (Name, dat)) in enumerate(zip(axes.flatten(),DataDict.items())):
        ax.plot(dat['TrainNb'],dat['MaxPower_pred'], label='Predicted')
        bl = dat['MaxPower_act'] != 0
        ax.plot(dat['TrainNb'][bl],dat['MaxPower_act'][bl], label='Actual')
        ax.legend()
        ax.set_title(Name, fontsize=14)
    fig.text(0.5, 0.04, 'Number of data points used for training', ha='center',fontsize=16)
    fig.text(0.04, 0.5, 'Power imparted to sample', va='center', rotation='vertical',fontsize=16)
    plt.savefig("{}/MaxPower.png".format(ResDir))
    plt.close()












        #
