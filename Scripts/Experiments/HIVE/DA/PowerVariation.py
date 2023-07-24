import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch

from Scripts.Common.ML import ML, GPR, NN

def GPR_compare(VL,DataDict):
    '''
    Function to compare the performance of GPR models with different kernels
    '''
    Parameters = DataDict['Parameters']

    MLModels = Parameters.MLModels
    TestData = Parameters.TestData

    TestIn, TestOut = ML.VLGetDataML(VL,TestData)
    TestMetric,TrainMetric = {}, {}
    kernels = []
    for model_name in MLModels:
        model_path = "{}/{}".format(VL.ML.output_dir,model_name)
        model = GPR.GetModel(model_path) # load in model
        kernel = model.ModelParameters['kernel']
        kernels.append(kernel)

        # analyse performance on train dtaa
        TrainIn,TrainOut = model.GetTrainData(to_numpy=True) #  get data used to generate model and convert to numpy
        pred = model.Predict(TrainIn)
        power_rmse,var_rmse = ML.RMSE(pred,TrainOut,axis=0) # calculate normalise root mean squared error
        TrainMetric[kernel] = [power_rmse,var_rmse] #  add to dictionary

        # analyse performance on test data
        pred = model.Predict(TestIn)
        power_rmse,var_rmse = ML.RMSE(pred,TestOut,axis=0)
        TestMetric[kernel] = [power_rmse,var_rmse] # add to dictionary

    fig, axes = plt.subplots(1,3,sharey=True,figsize=(15,5))
    x_point = []
    for _i,kernel in enumerate(kernels):
        i = 5*_i
        train_metric = TrainMetric[kernel]
        test_metric = TestMetric[kernel]
        # plot combined score
        axes[0].scatter([i],[np.mean(train_metric)],marker='x',c='k')
        axes[0].scatter([i+1],[np.mean(test_metric)],marker='o',c='k')
        # plot metrics for power prediction
        train_scatter = axes[1].scatter([i],[train_metric[0]],marker='x',c='k')
        test_scatter =  axes[1].scatter([i+1],[test_metric[0]],marker='o',c='k')
        # plot metrics for variation
        axes[2].scatter([i],[train_metric[1]],marker='x',c='k')
        axes[2].scatter([i+1],[test_metric[1]],marker='o',c='k')

        x_point.append(i+0.5)

    plt.setp(axes, xticks=x_point, xticklabels=kernels)
    axes[0].set_title('Average')
    axes[1].set_title('Power')
    axes[2].set_title('Variation')
    axes[1].legend([train_scatter,test_scatter], ['Train','Test'])
    plt.savefig("{}/GPR.png".format(DataDict['CALC_DIR']))
    plt.close()

def MLP_compare(VL,DataDict):
    '''
    Function to compare the performance of MLP models with different architecture
    '''
    Parameters = DataDict['Parameters']

    MLModels = Parameters.MLModels
    TestData = Parameters.TestData

    TestIn, TestOut = ML.VLGetDataML(VL,TestData)
    TestMetric,TrainMetric = {}, {}
    Architectures = []
    for model_name in MLModels:
        model_path = "{}/{}".format(VL.ML.output_dir,model_name)
        model = NN.GetModel(model_path) # load in model
        architecture = model.ModelParameters['Architecture']
        arch_str = '_'.join(map(str,architecture))
        Architectures.append(arch_str)

        # analyse performance on train dtaa
        TrainIn,TrainOut = model.GetTrainData(to_numpy=True) #  get data used to generate model and convert to numpy
        pred = model.Predict(TrainIn)
        power_rmse,var_rmse = ML.RMSE(pred,TrainOut,axis=0) # calculate normalise root mean squared error
        TrainMetric[arch_str] = [power_rmse,var_rmse] #  add to dictionary

        # analyse performance on test data
        pred = model.Predict(TestIn)
        power_rmse,var_rmse = ML.RMSE(pred,TestOut,axis=0)
        TestMetric[arch_str] = [power_rmse,var_rmse] # add to dictionary

    fig, axes = plt.subplots(1,3,sharey=True,figsize=(15,5))
    x_point = []
    for _i,arch_str in enumerate(Architectures):
        i = 5*_i
        train_metric = TrainMetric[arch_str]
        test_metric = TestMetric[arch_str]
        # plot combined score
        axes[0].scatter([i],[np.mean(train_metric)],marker='x',c='k')
        axes[0].scatter([i+1],[np.mean(test_metric)],marker='o',c='k')
        # plot metrics for power prediction
        train_scatter = axes[1].scatter([i],[train_metric[0]],marker='x',c='k')
        test_scatter =  axes[1].scatter([i+1],[test_metric[0]],marker='o',c='k')
        # plot metrics for variation
        axes[2].scatter([i],[train_metric[1]],marker='x',c='k')
        axes[2].scatter([i+1],[test_metric[1]],marker='o',c='k')

        x_point.append(i+0.5)

    plt.setp(axes, xticks=x_point, xticklabels=Architectures)
    axes[0].set_title('Average')
    axes[1].set_title('Power')
    axes[2].set_title('Variation')
    axes[1].legend([train_scatter,test_scatter], ['Train','Test'])
    plt.savefig("{}/MLP.png".format(DataDict['CALC_DIR']))
    plt.close()


