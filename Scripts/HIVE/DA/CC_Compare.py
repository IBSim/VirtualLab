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

from Functions import DataScale, DataRescale
from CoilConfig_GPR import ExactGPmodel, MSE
from CoilConfig_NN import NetPU

def Single(VL, MLdict):
    Parameters = MLdict["Parameters"]

    ResDir = "{}/{}".format(VL.PROJECT_DIR, MLdict["Name"])

    # Methods we are considering
    Methods = ['Grid','Random','Halton','Adaptive']
            #,'Adaptive_Uniform','Coupled_RBF','Coupled_Matern_1.5','Coupled_Matern_2.5']
    # Methods = ['Halton','Coupled_RBF']

    # Get TestNb number of points from TestData
    DataFile = "{}/ML/Data.hdf5".format(VL.PROJECT_DIR)
    MLData = h5py.File(DataFile,'r')
    TestData = MLData["TestData/PU_3"][:Parameters.TestNb].astype('float32')

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
            DataDict[Method] = {'TrainNb':[],'TestMSE':[],'TrainMSE':[],
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

        if os.path.isfile("{}/Data.pkl".format(ResSubDir)):
            with open("{}/Data.pkl".format(ResSubDir),'rb') as f:
                Data = pickle.load(f)
        else: Data = {}

        if MLParameters.File == 'CoilConfig_GPR':
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

            DataDict[Method]['TrainNb'].append(MLParameters.TrainNb)
            DataDict[Method]['TestMSE'].append([Test_MSE_P,Test_MSE_V])
            DataDict[Method]['TrainMSE'].append([Train_MSE_P,Train_MSE_V])

        else:

            model = NetPU(MLParameters.NNLayout,MLParameters.Dropout)
            model.load_state_dict(torch.load('{}/model.h5'.format(ResSubDir)))

            with torch.no_grad():
                Train_Vals = model(Train_x_tf).numpy()
                Test_Vals = model(Test_x_tf).numpy()

            Test_MSE_P = MSE(Test_Vals[:,0],Test_y_scale[:,0])
            Train_MSE_P = MSE(Train_Vals[:,0],Train_y_scale[:,0])
            Test_MSE_V = MSE(Test_Vals[:,1],Test_y_scale[:,1])
            Train_MSE_V = MSE(Train_Vals[:,1],Train_y_scale[:,1])
            if MLParameters.TrainNb in DataDict[Method]['TrainNb']:
                ix = DataDict[Method]['TrainNb'].index(MLParameters.TrainNb)
                s = sum(DataDict[Method]['TestMSE'][ix])
                if Test_MSE_P+Test_MSE_V < s:
                    DataDict[Method]['TestMSE'][ix] = [Test_MSE_P,Test_MSE_V]
                    DataDict[Method]['TrainMSE'][ix] = [Train_MSE_P,Train_MSE_V]
            else:
                DataDict[Method]['TrainNb'].append(MLParameters.TrainNb)
                DataDict[Method]['TestMSE'].append([Test_MSE_P,Test_MSE_V])
                DataDict[Method]['TrainMSE'].append([Train_MSE_P,Train_MSE_V])


        # Power_y = Data['MaxPower']['y']
        # if 'target' in Data['MaxPower']:
        #     Power_target = Data['MaxPower']['target']
        # elif os.path.isfile('{}/MaxPower.rmed'.format(ResSubDir)):
        #
        #     ERMESres = h5py.File('{}/MaxPower.rmed'.format(ResSubDir), 'r')
        #     attrs =  ERMESres["EM_Load"].attrs
        #     Scale = (1000/attrs['Current'])**2
        #     Watts = ERMESres["EM_Load/Watts"][:]*Scale
        #     JHNode =  ERMESres["EM_Load/JHNode"][:]*Scale
        #     ERMESres.close()
        #     Power_target = np.sum(Watts)
        # else:
        #     Power_target = 0
        #
        # DataDict[Method]['MaxPower_pred'].append(Power_y)
        # DataDict[Method]['MaxPower_act'].append(Power_target)


    for Name in Methods:
        if Name not in DataDict: continue
        dat = DataDict[Name]
        # if Name=='Coupled_RBF': Name = 'Coupled Halton'
        # Get sort order and arrange loss values accordingly
        srt = np.argsort(dat['TrainNb'])
        for key in dat.keys():
            if dat[key]:
                dat[key] = np.array(dat[key])[srt]
                print(key,dat[key])

    monochrome = (cycler('color', ['k']) * cycler('marker', [',', '^', '.']) * cycler('linestyle', ['-', '--', '-.',':']))
    for tp in ['Test','Train']:
        for i,res in enumerate(['Power','Variation']):
            fig, axes = plt.subplots(nrows=1,ncols=1, sharex=True,figsize=(15,10))
            axes.set_prop_cycle(monochrome)
            # fig.suptitle('MSE for {} based on different sampling methods\n and training dataset sizes'.format(res),fontsize=18)

            for Name in Methods:
                if Name not in DataDict: continue
                dat = DataDict[Name]
                # print(dat['{}MSE'.format(tp)][:,i]*OutputScaler[1,i]**2)
                axes.plot(dat['TrainNb'], dat['{}MSE'.format(tp)][:,i]*OutputScaler[1,i]**2,label=Name)
                # plt.show()
                axes.set_ylabel('MSE',fontsize=16)
                axes.set_xlabel('Number of points used for training',fontsize=16)
                axes.legend(fontsize=16)

            plt.savefig("{}/MSE_{}_{}.png".format(ResDir,res,tp))
            plt.close()

    # nc = 1
    # nr = int(np.ceil(len(DataDict)/nc))
    # fig, axes = plt.subplots(nrows=nr,ncols=nc, sharex=True,sharey=True, figsize=(15,10))
    # if nr==1:axes=np.array([axes])
    # i=0
    # for Name in Methods:
    #     if Name not in DataDict: continue
    #     dat = DataDict[Name]
    #     if Name=='Coupled_RBF': Name = 'Coupled Halton'
    #     ax = axes.flatten()[i]
    #     ax.plot(dat['TrainNb'],dat['MaxPower_pred'], label='Predicted')
    #     bl = dat['MaxPower_act'] != 0
    #     ax.plot(dat['TrainNb'][bl],dat['MaxPower_act'][bl], label='Actual')
    #     ax.legend()
    #     ax.set_title(Name, fontsize=14)
    #     i+=1
    # fig.text(0.5, 0.04, 'Number of data points used for training', ha='center',fontsize=16)
    # fig.text(0.04, 0.5, 'Power imparted to sample', va='center', rotation='vertical',fontsize=16)
    # plt.savefig("{}/MaxPower.png".format(ResDir))
    # plt.close()












        #
