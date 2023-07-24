#!/usr/bin/env python3
'''
This script demonstrates how the data collected in #SCRIPTNAME
can be used to weigh up the trade off between the power delivered to 
the sample and the uniformity of the heating profile.


'''

import sys
sys.dont_write_bytecode=True
from types import SimpleNamespace as Namespace
import numpy as np
from Scripts.Common.VirtualLab import VLSetup


CoilType='Pancake' # this can be 'Pancake' or 'HIVE'
CreateGPR = True
AnalyseGPR = False


# ====================================================================
# Setup VirtualLab
VirtualLab=VLSetup('HIVE','ML_analysis')

VirtualLab.Settings(Launcher='process',NbJobs=2)

# check data has been created, if not download
DataFile = '{}_coil/PowerVariation.hdf'.format(CoilType)
if not VirtualLab.InProject(DataFile):
    pass # Download data from somewhere

# ====================================================================
# Define GPR parameters to create model

ML = Namespace()
ML.File = ('GPR_Models','GPR_hdf5')
ML.TrainingParameters = {'Epochs':1000,'lr':0.05,'Print':50}
ML.TrainData = [DataFile, 'Features', [['Power'],['Variation']],{'group':'Train'}]
ML.TestData = [DataFile, 'Features', [['Power'],['Variation']],{'group':'Test'}]
main_parameters = Namespace(ML=ML)

ML = Namespace(Name = [],ModelParameters=[])
GPR_kernels = ['RBF','Matern_0.5','Matern_1.5','Matern_2.5']
for kernel in GPR_kernels:
    ML.ModelParameters.append({'kernel':kernel})
    ML.Name.append("PV/GPR/{}".format(kernel))

var_parameters = Namespace(ML=ML) 

VirtualLab.Parameters(main_parameters,var_parameters,RunML=CreateGPR)

# generate GPR models
VirtualLab.ML()

# ====================================================================
# analyse performance of GPR model

DA = Namespace()
DA.Name = "Analysis/PowerVariation"
DA.File = ['PowerVariation','GPR_compare']
DA.MLModels = var_parameters.ML.Name # use the models defined earlier
DA.TestData = [DataFile, 'Features', [['Power'],['Variation']],{'group':'Test'}] # train data is stored with the model, but test data is not

main_parameters = Namespace(DA=DA)

VirtualLab.Parameters(main_parameters,RunDA=AnalyseGPR)

# analyse GPR models
VirtualLab.DA()





