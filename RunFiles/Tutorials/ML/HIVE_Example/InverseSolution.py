#!/usr/bin/env python3
'''
This script demonstrates how the data collected in DataCollect.py
can be used to create 3D surrogate models of the temperature and 
Von Mises stress fields, which can be used to identify inverse solutions. 
'''

import sys
sys.dont_write_bytecode=True
from types import SimpleNamespace as Namespace
import numpy as np
from Scripts.Common.VirtualLab import VLSetup


CoilType='Pancake' # this can be 'Pancake' or 'HIVE'
CreateGPR = False
AnalyseGPR = False
CreateMLP = False
AnalyseMLP = True

# ====================================================================
# Setup VirtualLab
VirtualLab=VLSetup('HIVE','ML_analysis')

VirtualLab.Settings(Launcher='sequential',NbJobs=1)

# check data has been created, if not download
DataFile_T = '{}_coil/TempNodal.hdf'.format(CoilType)
if not VirtualLab.InProject(DataFile_T):
    pass # Download data from somewhere

# ====================================================================
# GPR analysis
# ====================================================================

# ====================================================================
# Define GPR parameters to create model

ML = Namespace()
ML.Name = 'Temperature/GPR'
ML.File = ('GPR_Models','GPR_PCA_hdf5')
ML.TrainingParameters = {'Epochs':1000,'lr':0.05,'Print':50}
ML.TrainData = [DataFile_T, 'Features', 'Temperature',{'group':'Train'}]
ML.ModelParameters = {'kernel':'Matern_2.5'} #{'min_noise':1e-8,'noise_init':1e-6}
ML.Metric = {'nb_components':10}

main_parameters = Namespace(ML=ML)

VirtualLab.Parameters(main_parameters,RunML=CreateGPR)

# generate GPR models
VirtualLab.ML()

# ====================================================================
# analyse performance of GPR model
DA = Namespace()
DA.Name = 'Analysis/InverseSolution/GPR_{}'.format(CoilType)
DA.File = ('InverseSolution','CreateImage_GPR')
DA.MLModel = 'Temperature/GPR'
# TODO: Download this mesh with data
DA.MeshName = 'HIVE_component' # name of the mesh used to generate the analysis (see DataCollect.py)
DA.TestData = [DataFile_T, 'Features', 'Temperature',{'group':'Test'}]
# create comparison plots for the following indexes of the test dataset. This can be any numbers up to 300 (the size of the test dataset)
DA.Index = [1,2]

main_parameters = Namespace(DA=DA)

VirtualLab.Parameters(main_parameters,RunDA=AnalyseGPR)

VirtualLab.DA()

# ====================================================================
# MLP analysis
# ====================================================================

# ====================================================================
# Define MLP parameters to create model

ML = Namespace()
ML.Name = 'Temperature/MLP'
ML.File = ('NN_Models','MLP_PCA_hdf5')
ML.TrainingParameters = {'Epochs':1000,'lr':0.005,'Print':50}
ML.TrainData = [DataFile_T, 'Features', 'Temperature',{'group':'Train'}]
ML.ValidationData = [DataFile_T, 'Features', 'Temperature',{'group':'Test'}]
ML.ModelParameters = {'Architecture':[8,16,8]} 
ML.Seed = 100 # initial weights of MLP are randomised so this ensures reproducability
ML.Metric = {'nb_components':10}
main_parameters = Namespace(ML=ML)

VirtualLab.Parameters(main_parameters,RunML=CreateMLP)

# generate GPR models
VirtualLab.ML()

# ====================================================================
# analyse performance of MLP model
DA = Namespace()
DA.Name = 'Analysis/InverseSolution/MLP_{}'.format(CoilType)
DA.File = ('InverseSolution','CreateImage_MLP')
DA.MLModel = 'Temperature/MLP'
# TODO: Download this mesh with data
DA.MeshName = 'HIVE_component' # name of the mesh used to generate the analysis (see DataCollect.py)
DA.TestData = [DataFile_T, 'Features', 'Temperature',{'group':'Test'}]
# create comparison plots for the following indexes of the test dataset. This can be any numbers up to 300 (the size of the test dataset)
DA.Index = [1,2]

main_parameters = Namespace(DA=DA)

VirtualLab.Parameters(main_parameters,RunDA=AnalyseMLP)

VirtualLab.DA()