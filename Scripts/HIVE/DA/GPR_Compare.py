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
from cycler import cycler
from natsort import natsorted

from Functions import DataScale, DataRescale
from CoilConfig_GPR import ExactGPmodel, MSE
from CoilConfig_NN import NetPU

def Single(VL, DADict):
    Parameters = DADict["Parameters"]

    ResDir = "{}/{}".format(VL.PROJECT_DIR, DADict["Name"])

    # Methods we are considering
    Methods = Parameters.Methods

    # Get TestNb number of points from TestData
    DataFile = "{}/ML/Data.hdf5".format(VL.PROJECT_DIR)
    MLData = h5py.File(DataFile,'r')
    TestData = MLData["TestData/PU_3"][:Parameters.TestNb].astype('float32')
    Test_x,Test_y = TestData[:,:4],TestData[:,4:]

    DataDict = {}
    for ResName in natsorted(os.listdir(ResDir)):
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

        if Method not in Methods: continue
        if hasattr(Parameters,'Range'):
            if MLParameters.TrainNb < Parameters.Range[0] \
                or MLParameters.TrainNb > Parameters.Range[1]: continue
        if hasattr(Parameters,'TrainNbs'):
            if MLParameters.TrainNb not in Parameters.TrainNbs:
                continue

        if Method not in DataDict:
            DataDict[Method] = {'TrainNb':[],'TestMSE':[],'TrainMSE':[],
                                'MaxPower_pred':[],'MaxPower_act':[],'MaxPower_data':[]}

        # Get data used to train model
        TrainData = (np.load("{}/TrainData.npy".format(ResSubDir))).astype('float32')
        Train_x,Train_y = TrainData[:,:4],TrainData[:,4:]
        DataDict[Method]['MaxPower_data'].append(Train_y.max(axis=0)[0])

        # Scale train and test data
        InputScaler = np.array([Train_x.min(axis=0),Train_x.max(axis=0) - Train_x.min(axis=0)])
        OutputScaler = np.array([Train_y.min(axis=0),Train_y.max(axis=0) - Train_y.min(axis=0)])

        Train_x_scale, Train_y_scale = DataScale(Train_x,*InputScaler), DataScale(Train_y,*OutputScaler)
        Test_x_scale,Test_y_scale = DataScale(Test_x,*InputScaler), DataScale(Test_y,*OutputScaler)
        Train_x_tf,Train_y_tf = torch.from_numpy(Train_x_scale),torch.from_numpy(Train_y_scale)
        Test_x_tf,Test_y_tf = torch.from_numpy(Test_x_scale),torch.from_numpy(Test_y_scale)

        if os.path.isfile("{}/Data.pkl".format(ResSubDir)):
            with open("{}/Data.pkl".format(ResSubDir),'rb') as f:
                Data = pickle.load(f)
        else: Data = {}

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
        # Variation model
        Variation = ExactGPmodel(Train_x_tf, Train_y_tf[:,1], VarLH,
                        MLParameters.Kernel,options)
        state_dict_V = torch.load('{}/Variation.pth'.format(ResSubDir))
        Variation.load_state_dict(state_dict_V)
        VarLH.eval(); Variation.eval()

        with torch.no_grad(), gpytorch.settings.max_cholesky_size(1500), gpytorch.settings.debug(False):
            Test_MSE_P = MSE(Power(Test_x_tf).mean.numpy(), Test_y_scale[:,0])
            Train_MSE_P = MSE(Power(Train_x_tf).mean.numpy(), Train_y_scale[:,0])
            Test_MSE_V = MSE(Variation(Test_x_tf).mean.numpy(), Test_y_scale[:,1])
            Train_MSE_V = MSE(Variation(Train_x_tf).mean.numpy(), Train_y_scale[:,1])
            print(Method,MLParameters.TrainNb,Test_MSE_P*OutputScaler[1,0]**2,Test_MSE_V*OutputScaler[1,1]**2)

        DataDict[Method]['TrainNb'].append(MLParameters.TrainNb)
        DataDict[Method]['TestMSE'].append([Test_MSE_P,Test_MSE_V])
        DataDict[Method]['TrainMSE'].append([Train_MSE_P,Train_MSE_V])

        Power_y = Data['MaxPower']['y']
        if 'target' in Data['MaxPower']:
            Power_target = Data['MaxPower']['target']
        elif os.path.isfile('{}/MaxPower.rmed'.format(ResSubDir)):
            ERMESres = h5py.File('{}/MaxPower.rmed'.format(ResSubDir), 'r')
            Volumes = ERMESres["EM_Load/Volumes"][:]
            JH_Vol = ERMESres["EM_Load/JH_Vol"][:]
            Watts = JH_Vol*Volumes
            ERMESres.close()
            Power_target = np.sum(Watts)
        else:
            Power_target = 0
        print(Power_target)
        DataDict[Method]['MaxPower_pred'].append(Power_y)
        DataDict[Method]['MaxPower_act'].append(Power_target)

    Methods = [i for i in Methods if i in DataDict] # Prserves order

    # monochrome = (cycler('color', ['k']) * cycler('marker', ['x']) * cycler('linestyle', ['-', '--',':', '-.']))
    colours = plt.cm.gray(np.linspace(0,0.6,len(Methods)))
    lim=[200,1000]
    fnt = 24
    for tp in ['Test','Train']:
        for i,res in enumerate(['Power','Variation']):
            nmh=''
            fig, axes = plt.subplots(nrows=1,ncols=1, sharex=True,figsize=(15,10))
            for  j, Name in enumerate(Methods):

                Nb = DataDict[Name]['TrainNb']
                mse = np.array(DataDict[Name]['{}MSE'.format(tp)])[:,i]
                if Name=='Adaptive': Name = 'PyAdaptive'
                if Name=='Hybrid_Halton': Name = 'HII-Halton'

                axes.plot(Nb, mse*OutputScaler[1,i]**2,markersize=15,marker='x',c=colours[j], label=Name)

            axes.set_ylabel('MSE',fontsize=fnt)
            axes.set_xlabel('Number of points used for training',fontsize=fnt)
            axes.set_yscale('log')
            axes.set_ylim(bottom=1,top=lim[i])
            axes.legend(fontsize=fnt)
            plt.xticks(fontsize=fnt)
            plt.yticks(fontsize=fnt)
            axes = plt.gca()
            axes.yaxis.grid()

            plt.savefig("{}/MSE_{}_{}{}.eps".format(ResDir,res,tp,nmh),dpi=600)
            plt.close()

            xlim = axes.get_xlim()

    # fig, axes = plt.subplots(nrows=1,ncols=1, figsize=(15,10))
    # marker = ['x','^','o']
    # for i, Name in enumerate(Methods):
    #     dat = DataDict[Name]
    #
    #     Nb = np.array(DataDict[Name]['TrainNb'])
    #     Pred = dat['MaxPower_pred']
    #     Act = np.array(dat['MaxPower_act'])
    #     bl = Act != 0
    #
    #     axes.plot(Nb,Pred, c='0',marker=marker[i],label=Name)
    #     axes.plot(Nb[bl],Act[bl], c='0', marker=marker[i],linestyle='--')
    #     axes.scatter(Nb,dat['MaxPower_data'],c='0',marker=marker[i])
    #     nmh = ''
    #
    #     axes.set_ylim([480,550])
    #     axes.set_ylabel('Max. Power',fontsize=20)
    #     axes.set_xlabel('Number of points used for training',fontsize=20)
    #     axes.legend(loc='upper right',fontsize=20)
    #     plt.xticks(fontsize=20)
    #     plt.yticks(fontsize=20)
    #     plt.grid()
    #
    # plt.savefig("{}/MaxPower{}.png".format(ResDir,nmh))
    # plt.close()

    fnt = 36
    for i, Name in enumerate(Methods):
        fig, axes = plt.subplots(nrows=1,ncols=1, figsize=(15,10))
        dat = DataDict[Name]

        if Name=='Adaptive': Name = 'PyAdaptive'

        Nb = np.array(dat['TrainNb'])
        Pred = dat['MaxPower_pred']
        Act = np.array(dat['MaxPower_act'])
        bl = Act != 0

        axes.scatter(Nb,Pred, marker='o', s=200, edgecolor='k',  facecolors='none', label='Predicted')
        axes.scatter(Nb[bl],Act[bl], marker='+', s=300, edgecolor='k',  facecolors='k', label='Actual')
        axes.scatter(Nb,dat['MaxPower_data'],s=200,c='0',marker='x', label='Max. Train')
        print(dat['MaxPower_data'])
        nmh = ''

        axes.set_xlim(xlim)
        axes.set_xlabel('Number of points used for training',fontsize=fnt)
        axes.set_ylim([480,540])
        axes.set_ylabel('Power',fontsize=fnt)

        # axes.legend(loc='upper right',fontsize=fnt)
        plt.xticks(fontsize=fnt)
        plt.yticks(fontsize=fnt)
        axes = plt.gca()
        axes.yaxis.grid()

        plt.savefig("{}/MaxPower_{}{}.eps".format(ResDir,Name,nmh),dpi=600)
        plt.close()












        #
