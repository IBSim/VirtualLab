#!/usr/bin/env python3
#===============================================================================
# Header
#===============================================================================

import sys
sys.dont_write_bytecode=True
from Scripts.Common.VirtualLab import VLSetup
from types import SimpleNamespace as Namespace
import numpy as np

def f_1D(x):
    """The 1D function to predict is x*sin(x)"""
    f = np.array(x) * np.sin(x)
    return f.tolist() 

#===============================================================================
# Environment
#===============================================================================

Simulation='Examples'
Project='ML_1D'

VirtualLab=VLSetup(
           Simulation,
           Project
           )

VirtualLab.Settings(
           Mode='T',
           Launcher='Process',
           NbJobs=1
           )

#===============================================================================
# Definitions & Parameters
#===============================================================================

ML = Namespace()
ML.Name = 'MLP_Example'
ML.File = ['NN_Models','MLP_data']
TrainInput = [1,4,5,8,9]
ML.TrainData = [TrainInput,f_1D(TrainInput)]
ValidationInput = [2,6,8.5]
ML.ValidationData = [ValidationInput,f_1D(ValidationInput)]
ML.ModelParameters = {'Architecture':[8,8]} # parameters of the model
ML.TrainingParameters = {'Epochs':500, 'lr':0.05, 'Print':20} # parameters to dictate raining the model
ML.Seed = 100

DA = Namespace()
DA.Name = 'Analysis/MLP_Example'
DA.File = ['MLAnalysis','MLP_analysis_1D']
DA.ModelName = 'MLP_Example'
DA.Range = [0,10]
DA.PlotName = 'MLP'

Parameters = Namespace(ML=ML,DA=DA)

VirtualLab.Parameters(
           Parameters,
           RunML=True,
           RunDA=True
           )

#===============================================================================
# Build Model
#===============================================================================

VirtualLab.ML()

#===============================================================================
# Analyse Model
#===============================================================================

VirtualLab.DA()
