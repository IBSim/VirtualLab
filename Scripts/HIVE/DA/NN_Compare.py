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
from natsort import natsorted


from Functions import DataScale, DataRescale
from CoilConfig_GPR import ExactGPmodel, MSE
from CoilConfig_NN import NetPU

def Single(VL, MLdict):
    Parameters = MLdict["Parameters"]

    ResDir = "{}/{}".format(VL.PROJECT_DIR, MLdict["Name"])

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
        if Method == 'HIVE_Random': Method = 'Random'
        if Method not in Methods: continue

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

        model = NetPU(MLParameters.NNLayout,MLParameters.Dropout)
        model.load_state_dict(torch.load('{}/model.h5'.format(ResSubDir)))

        with torch.no_grad():
            Train_Vals = model(Train_x_tf).numpy()
            Test_Vals = model(Test_x_tf).numpy()

        Test_MSE_P = MSE(Test_Vals[:,0],Test_y_scale[:,0])
        Train_MSE_P = MSE(Train_Vals[:,0],Train_y_scale[:,0])
        Test_MSE_V = MSE(Test_Vals[:,1],Test_y_scale[:,1])
        Train_MSE_V = MSE(Train_Vals[:,1],Train_y_scale[:,1])

        DataDict[Method]['TrainNb'].append(MLParameters.TrainNb)
        DataDict[Method]['TestMSE'].append([Test_MSE_P,Test_MSE_V])
        DataDict[Method]['TrainMSE'].append([Train_MSE_P,Train_MSE_V])


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

    Methods = [i for i in Methods if i in DataDict] # Prserves order
    colours = plt.cm.gray(np.linspace(0,0.6,len(Methods)))
    # colours = plt.cm.brg(np.linspace(0.1,0.9,len(Methods)))
    lim = [200,1000]
    fnt = 24
    for tp in ['Test','Train']:
        for i,res in enumerate(['Power','Variation']):
            fig, axes = plt.subplots(nrows=1,ncols=1, sharex=True,figsize=(15,10))

            for  j, Name in enumerate(Methods):
                Nb = DataDict[Name]['TrainNb']
                Nb_Uni = np.unique(Nb)
                mse = np.array(DataDict[Name]['{}MSE'.format(tp)])[:,i]
                mean,min,max = [],[],[]
                for v in Nb_Uni:
                    vals = mse[Nb==v]
                    _mean = np.mean(vals)
                    max.append(vals.max())
                    min.append(vals.min())
                    mean.append(_mean)
                    # err.append([_mean-vals.min(),vals.max()-_mean])

                mean= np.array(mean)*OutputScaler[1,i]**2
                min = np.array(min)*OutputScaler[1,i]**2
                max = np.array(max)*OutputScaler[1,i]**2
                axes.plot(Nb_Uni, mean,markersize=15, marker='x',c=colours[j], label=Name)
                # axes.plot(Nb_Uni, min,c=colours[j],linestyle='--',alpha=0.3)
                # axes.plot(Nb_Uni, max,c=colours[j],linestyle='--',alpha=0.3)
                # plt.show()
            axes.set_ylabel('MSE',fontsize=fnt)
            axes.set_xlabel('Number of points used for training',fontsize=fnt)
            axes.legend(fontsize=fnt)
            axes.set_yscale('log')
            axes.set_ylim(bottom=1,top=lim[i])
            plt.xticks(fontsize=fnt)
            plt.yticks(fontsize=fnt)
            plt.grid()

            plt.savefig("{}/MSE_{}_{}.png".format(ResDir,res,tp))
            plt.close()

    fig, axes = plt.subplots(nrows=1,ncols=1,figsize=(15,10))
    Nb = DataDict['Halton']['TrainNb']
    Nb_Uni = np.unique(Nb)
    mse = np.array(DataDict['Halton']['TestMSE'])[:,0]
    ex1 = [mse[Nb==v][0] for v in Nb_Uni]
    ex1 = np.array(ex1)*OutputScaler[1,0]**2
    ex2 = [mse[Nb==v][8] for v in Nb_Uni]
    ex2 = np.array(ex2)*OutputScaler[1,0]**2
    axes.plot(Nb_Uni,ex1,c='0',label='Halton_0')
    axes.plot(Nb_Uni,ex2,c='0.5',label='Halton_1')
    axes.set_ylabel('MSE',fontsize=20)
    axes.set_xlabel('Number of points used for training',fontsize=20)
    axes.legend(fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig("{}/MSE_Power_Seed.png".format(ResDir))

    plt.close()
    fnt = 36
    for i, Name in enumerate(Methods):
        fig, axes = plt.subplots(nrows=1,ncols=1, figsize=(15,10))
        dat = DataDict[Name]
        # if Name=='Coupled_RBF': Name = 'Coupled Halton'

        _Pred = np.array(dat['MaxPower_pred'])[:,0]
        _Act = np.array(dat['MaxPower_act'])
        _MPData = np.array(dat['MaxPower_data'])
        Nb = np.array(DataDict[Name]['TrainNb'])
        Nb_Uni = np.unique(Nb)
        msesum = np.array(dat['TestMSE']).sum(axis=1)
        Pred,Act,MPData = [],[],[]
        for v in Nb_Uni:
            bl = Nb==v
            vals = msesum[bl]
            ix = np.argmin(vals)

            Pred.append(_Pred[bl][ix])
            Act.append(_Act[bl][ix])
            MPData.append(_MPData[bl][0])

        Pred,Act = np.array(Pred),np.array(Act)

        axes.plot(Nb_Uni,Pred, c='0',label='Predicted')
        bl = Act != 0
        axes.plot(Nb_Uni[bl],Act[bl], c='0', linestyle='--',label='Actual')

        axes.scatter(Nb_Uni,MPData,s=200,c='0',marker='x', label='Max. Train')
        axes.set_ylim([480,550])
        axes.set_ylabel('Max. Power',fontsize=fnt)
        axes.set_xlabel('Number of points used for training',fontsize=fnt)
        axes.legend(fontsize=fnt)
        plt.xticks(fontsize=fnt)
        plt.yticks(fontsize=fnt)
        plt.grid()

        plt.savefig("{}/MaxPower_{}.png".format(ResDir,Name))
        plt.close()













        #
