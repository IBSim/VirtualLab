#!/usr/bin/env python3
'''
This script demonstrates how the data collected in DataCollect.py
can be used to create 3D surrogate models of the temperature and 
Von Mises stress fields, which can be used to identify inverse solutions.

This script assumes that the analysis in InverseSolution.py has already been performed
'''

import sys
sys.dont_write_bytecode=True
from types import SimpleNamespace as Namespace
import numpy as np
from Scripts.Common.VirtualLab import VLSetup


CoilType='Pancake' # this can be 'Pancake' or 'HIVE'
PCA_Analysis = True
CreateGPR = False
AnalyseGPR = False
CreateMLP = False
AnalyseMLP = False

# ====================================================================
# Setup VirtualLab
VirtualLab=VLSetup('HIVE','ML_analysis')

VirtualLab.Settings(Launcher='sequential',NbJobs=1)

# check data has been created, if not download
DataFile = '{}_coil/VMNodal.hdf'.format(CoilType)
if not VirtualLab.InProject(DataFile):
    pass # Download data from somewhere (and possibly mesh)

DA = Namespace()
DA.Name = 'Analysis/{}/InverseSolution_VM/PCA_Sensitivity'.format(CoilType)
DA.File = ('MLtools','PCA_Sensitivity')
DA.TrainData = [DataFile, 'Features', 'VonMises',{'group':'Train'}]
DA.TestData = [DataFile, 'Features', 'VonMises',{'group':'Test'}]

main_parameters = Namespace(DA=DA)

VirtualLab.Parameters(main_parameters,RunDA=PCA_Analysis)

VirtualLab.DA()

# ====================================================================
# GPR analysis
# ====================================================================

# ====================================================================
# Define GPR parameters to create model

ML = Namespace()
ML.Name = 'VonMises/GPR'
ML.File = ('GPR_Models','GPR_PCA_hdf5')
ML.TrainingParameters = {'Epochs':1000,'lr':0.05,'Print':50}
ML.TrainData = [DataFile, 'Features', 'VonMises',{'group':'Train'}]
ML.ModelParameters = {'kernel':'Matern_2.5'} #{'min_noise':1e-8,'noise_init':1e-6}
ML.Metric = {'nb_components':10}

main_parameters = Namespace(ML=ML)

VirtualLab.Parameters(main_parameters,RunML=CreateGPR)

# generate GPR models
VirtualLab.ML()

# ====================================================================
# analyse performance of GPR model
DA = Namespace()
DA.Name = 'Analysis/{}/InverseSolution_VM/GPR'.format(CoilType)
DA.File = ('InverseSolution','AnalysisVM_GPR')
DA.MLModel_T = 'Temperature/GPR'
DA.MLModel_VM = 'VonMises/GPR'
DA.MeshName = 'HIVE_component' # name of the mesh used to generate the analysis (see DataCollect.py)
DA.TestData = [DataFile, 'Features', 'VonMises',{'group':'Test'}]
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
ML.Name = 'VonMises/MLP'
ML.File = ('NN_Models','MLP_PCA_hdf5')
ML.TrainingParameters = {'Epochs':1000,'lr':0.005,'Print':50}
ML.TrainData = [DataFile, 'Features', 'VonMises',{'group':'Train'}]
ML.ValidationData = [DataFile, 'Features', 'VonMises',{'group':'Test'}]
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
DA.Name = 'Analysis/{}/InverseSolution_VM/MLP'.format(CoilType)
DA.File = ('InverseSolution','AnalysisVM_MLP')
DA.MLModel_T = 'Temperature/MLP'
DA.MLModel_VM = 'VonMises/MLP'
DA.MeshName = 'HIVE_component' # name of the mesh used to generate the analysis (see DataCollect.py)
DA.TestData = [DataFile, 'Features', 'VonMises',{'group':'Test'}]
# create comparison plots for the following indexes of the test dataset. This can be any numbers up to 300 (the size of the test dataset)
DA.Index = [1,2]

main_parameters = Namespace(DA=DA)

VirtualLab.Parameters(main_parameters,RunDA=AnalyseMLP)

VirtualLab.DA()