import os
import sys

import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt

from Scripts.Common.ML import ML, GPR

'''
Example script for how to create Gaussian Process regression (GPR) models in
VirtualLab.
'''


def f_1D(x):
    """The 1D function to predict."""
    return x * np.sin(x)

def Example_1D(VL,DADict):
    Parameters = DADict['Parameters']
    np.random.seed(100)

    lim = getattr(Parameters,'Limits',[0,10])
    # Training data
    TrainIn = np.random.uniform(*lim,Parameters.NbTrain)
    TrainOut = f_1D(TrainIn)
    # Testing data
    TestIn = np.random.uniform(*lim,Parameters.NbTest)
    TestOut = f_1D(TestIn)

    # ==========================================================================
    # Get parameters and build model
    # Parameters relating to the model
    ModelParameters = getattr(Parameters,'ModelParameters',{})
    # Parameters relating to the training
    TrainingParameters = getattr(Parameters,'TrainingParameters',{})

    # Build model. this returns the likelihood function, the model, and a Dataspace,
    # which holds all of the data and scalings used (all input and outptu data is
    # scaled to [0,1] range).
    likelihood, model, Dataspace = GPR.BuildModel([TrainIn,TrainOut],
                            DADict['CALC_DIR'], ModelParameters=ModelParameters,
                            TrainingParameters=TrainingParameters)


    # make plot showing model v true function
    x_linspace = np.linspace(*lim,100)
    x_linspace_scale = ML.DataScale(x_linspace,*Dataspace.InputScaler) # scale down to [0,1] range
    x_linspace_scale = torch.from_numpy(x_linspace_scale) # convert to pytorch tensor
    with torch.no_grad():
        model_pred = model(x_linspace_scale)
    # get mean and variance & convert to numpy array (from pytorch tensor)
    mean = model_pred.mean.numpy()
    stddev = model_pred.stddev.numpy()
    # rescale up to true values. Variance is scaled differently to the mean.
    mean = ML.DataRescale(mean,*Dataspace.OutputScaler)
    stddev = ML.DataRescale(stddev,0,Dataspace.OutputScaler[1])

    plt.figure()
    plt.plot(x_linspace,mean,label='GPR Model')
    plt.scatter(TrainIn,TrainOut,c='g',label='Training points')
    plt.fill(np.concatenate([x_linspace,x_linspace[::-1]]),
             np.concatenate([mean - 1.96*stddev, (mean + 1.96*stddev)[::-1]]),
             alpha=.2, fc='b', ec='None', label='95% confidence interval')
    plt.plot(x_linspace,f_1D(x_linspace),label='True function')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('f')
    plt.show()

    # ==========================================================================
    #  get error metrics of models

    # add test data to namespace so its scaled accordingly
    ML.DataspaceAdd(Dataspace,Test=[TestIn,TestOut])
    # make predictions on test and training data
    with torch.no_grad():
        pred_train = model(Dataspace.TrainIn_scale)
        pred_test = model(Dataspace.TestIn_scale)
    train_mean = pred_train.mean.numpy()
    train_mean = ML.DataRescale(train_mean,*Dataspace.OutputScaler)
    test_mean = pred_test.mean.numpy()
    test_mean = ML.DataRescale(test_mean,*Dataspace.OutputScaler)
    # compare answers with true values
    train_metrics = ML.GetMetrics2(train_mean,TrainOut)
    print("\nTraining metrics:\n{}\n".format(train_metrics))
    test_metrics = ML.GetMetrics2(test_mean,TestOut)
    print("Testing metrics:\n{}\n".format(test_metrics))
