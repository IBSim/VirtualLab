#!/usr/bin/env python3
'''
This script demonstrates how the data collected in DataCollect.py
can be used to predict the heating profile on the surface 
adjacent to the induction coil.
'''

import sys
sys.dont_write_bytecode=True
from types import SimpleNamespace as Namespace
import numpy as np
from Scripts.Common.VirtualLab import VLSetup


CoilType='Pancake' # this can be 'Pancake' or 'HIVE'
CreateGPR = False
AnalyseGPR = True
CreateMLP = False
AnalyseMLP = True


# ====================================================================
# Setup VirtualLab
VirtualLab=VLSetup('HIVE','ML_analysis')

VirtualLab.Settings(Launcher='sequential',NbJobs=1)

# check data has been created, if not download
DataFile = '{}_coil/JouleHeating.hdf'.format(CoilType)
if not VirtualLab.InProject(DataFile):
    pass # Download data from somewhere

# ====================================================================
# GPR analysis
# ====================================================================

# ====================================================================
# Define GPR parameters to create model

ML = Namespace()
ML.Name = 'HeatProfile/GPR'
ML.File = ('GPR_Models','GPR_PCA_hdf5')
ML.TrainingParameters = {'Epochs':1000,'lr':0.05,'Print':50}
ML.TrainData = [DataFile, 'Features', 'SurfaceJH',{'group':'Train'}]
ML.ModelParameters = {'kernel':'Matern_2.5'} #{'min_noise':1e-8,'noise_init':1e-6}
ML.Metric = {'nb_components':10}

main_parameters = Namespace(ML=ML)

VirtualLab.Parameters(main_parameters,RunML=CreateGPR)

# generate GPR models
VirtualLab.ML()

# ====================================================================
# analyse performance of GPR model
DA = Namespace()
DA.Name = 'Analysis/HeatingProfile/GPR_{}'.format(CoilType)
DA.File = ('HeatingProfile','CreateImage_GPR')
DA.MLModel = 'HeatProfile/GPR'
# TODO: Download this mesh with data
DA.MeshName = 'HIVE_component' # name of the mesh used to generate the analysis (see DataCollect.py)
DA.TestData = [DataFile, 'Features', 'SurfaceJH',{'group':'Test'}]
DA.Index = [1]


main_parameters = Namespace(DA=DA)

VirtualLab.Parameters(main_parameters)

VirtualLab.DA()


# ====================================================================
# MLP analysis
# ====================================================================

# ====================================================================
# Define MLP parameters to create model