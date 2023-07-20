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
# Definitions & Parameters
#===============================================================================

Simulation='Examples'
Project='ML_1D'

ML = Namespace()
ML.Name = 'GPR_Example'
ML.File = ['GPR_Models','GPR_data']
ML.TrainInputData = [1,4,5,8,9]
ML.TrainOutputData = f_1D(ML.TrainInputData)
ML.ModelParameters = {'kernel':'RBF'} # parameters of the model
ML.TrainingParameters = {'Epochs':500, 'lr':0.05, 'Print':20} # parameters to dictate raining the model
ML.PrintParameters = True

DA = Namespace()
DA.Name = 'Analysis/GPR_Example'
DA.File = ['MLAnalysis','GPR_analysis_1D']
DA.ModelName = 'GPR_Example'
DA.Range = [0,10]
DA.PlotName = 'GPR'

Parameters = Namespace(ML=ML,DA=DA)

#===============================================================================
# Environment
#===============================================================================

VirtualLab=VLSetup(
           Simulation,
           Project
           )

VirtualLab.Settings(
           Mode='T',
           Launcher='Process',
           NbJobs=1
           )

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
