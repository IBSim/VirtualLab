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
    """The 1D function to predict is x*sin(x)"""
    f = np.array(x) * np.sin(x)
    return f.tolist() 


def GPR_analysis_1D(VL,DataDict):
    Parameters = DataDict['Parameters']
    model_name = Parameters.ModelName
    plot_range = Parameters.Range
    plot_name = Parameters.PlotName

    model_dir = "{}/{}".format(VL.ML.output_dir,model_name)

    mod = GPR.GetModel(model_dir)

    x = np.linspace(*plot_range,100)
    y_mod, y_conf = mod.Predict(x, return_confidence=True)
    y_true = f_1D(x)
    train_x,train_y = mod.GetTrainData(to_numpy=True)

    plt.figure()
    plt.scatter(train_x,train_y,marker='x')
    plt.plot(x,y_true,label='x * sin(x)',c='g')
    plt.plot(x,y_mod,label='Model',c='b')
    plt.fill(np.concatenate([x,x[::-1]]),
             np.concatenate([y_mod - 1.96*y_conf, (y_mod + 1.96*y_conf)[::-1]]),
             alpha=.2, fc='b', ec='None', label='95% CI')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('f')
    plt.savefig("{}/{}.png".format(DataDict['CALC_DIR'],plot_name))
    plt.close()





