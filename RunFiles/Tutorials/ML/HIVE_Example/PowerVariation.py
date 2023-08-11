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
CreateGPR = False
AnalyseGPR = True
CreateMLP = False
AnalyseMLP = True

# ====================================================================
# Setup VirtualLab
VirtualLab=VLSetup('HIVE','ML_analysis')

VirtualLab.Settings(Launcher='sequential',NbJobs=1)

# check data has been created, if not download
DataFile = '{}_coil/PowerVariation.hdf'.format(CoilType)
if not VirtualLab.InProject(DataFile):
    pass # Download data from somewhere

# ====================================================================
# GPR analysis
# ====================================================================

# ====================================================================
# Define GPR parameters to create model

ML = Namespace()
ML.File = ('GPR_Models','GPR_hdf5')
ML.TrainingParameters = {'Epochs':1000,'lr':0.05,'Print':50}
ML.TrainData = [DataFile, 'Features', [['Power'],['Variation']],{'group':'Train'}]
main_parameters = Namespace(ML=ML)

ML = Namespace(Name = [],ModelParameters=[])
GPR_kernels = ['RBF','Matern_0.5','Matern_1.5','Matern_2.5']
for kernel in GPR_kernels:
    ML.ModelParameters.append({'kernel':kernel})
    ML.Name.append("PV/{}/GPR/{}".format(CoilType,kernel))

var_parameters = Namespace(ML=ML) 

VirtualLab.Parameters(main_parameters,var_parameters,RunML=CreateGPR)

# generate GPR models
VirtualLab.ML()

# ====================================================================
# analyse performance of GPR model

DA = Namespace()
DA.Name = "Analysis/{}/PowerVariation/GPR".format(CoilType)
DA.File = ['PowerVariation','GPR_compare']
DA.MLModels = var_parameters.ML.Name # use the models defined earlier
DA.TestData = [DataFile, 'Features', [['Power'],['Variation']],{'group':'Test'}] # unseen data to analyse performance

main_parameters = Namespace(DA=DA)

VirtualLab.Parameters(main_parameters,RunDA=AnalyseGPR)

VirtualLab.DA()

# ====================================================================
# create performance envelope

DA = Namespace()
DA.Name = "Analysis/{}/PowerVariation/GPR".format(CoilType)
DA.File = ['PowerVariation','Insight_GPR']
DA.MLModel = "PV/{}/GPR/RBF".format(CoilType) # chose a single model to gain insight from
main_parameters = Namespace(DA=DA)

VirtualLab.Parameters(main_parameters,RunDA=AnalyseGPR)

VirtualLab.DA()


# ====================================================================
# MLP analysis
# ====================================================================

# ====================================================================
# Define MLP parameters to create model

ML = Namespace()
ML.File = ('NN_Models','MLP_hdf5')
ML.TrainingParameters = {'Epochs':1000,'lr':0.05,'Print':50}
ML.TrainData = [DataFile, 'Features', [['Power'],['Variation']],{'group':'Train'}]
ML.ValidationData = [DataFile, 'Features', [['Power'],['Variation']],{'group':'Test'}] # data used to monitor for overfitting
ML.Seed = 100 # initial weights of MLP are randomised so this ensures reproducability
main_parameters = Namespace(ML=ML)

ML = Namespace(Name = [],ModelParameters=[])
Architectures = [[32,32],[16,32,16],[8,16,8,4]] # the hidden layers of the MLP
for architecture in Architectures:
    ML.ModelParameters.append({'Architecture':architecture})
    arch_str = '_'.join(map(str,architecture)) # convert architecture to string and save under that name
    ML.Name.append("PV/{}/MLP/{}".format(CoilType,arch_str))

var_parameters = Namespace(ML=ML) 

VirtualLab.Parameters(main_parameters,var_parameters,RunML=CreateMLP)

# generate MLP models
VirtualLab.ML()


# ====================================================================
# analyse performance of MLP model

DA = Namespace()
DA.Name = "Analysis/{}/PowerVariation/MLP".format(CoilType) # results will be saved to same directory as before
DA.File = ['PowerVariation','MLP_compare']
DA.MLModels = var_parameters.ML.Name # use the models defined earlier
DA.TestData = [DataFile, 'Features', [['Power'],['Variation']],{'group':'Test'}] # unseen data to analyse performance

main_parameters = Namespace(DA=DA)

VirtualLab.Parameters(main_parameters,RunDA=AnalyseMLP)

VirtualLab.DA()

# ====================================================================
# create performance envelope

DA = Namespace()
DA.Name = "Analysis/{}/PowerVariation/MLP".format(CoilType)
DA.File = ['PowerVariation','Insight_MLP']
DA.MLModel = "PV/{}/MLP/32_32".format(CoilType) # chose a single model to gain insight from
main_parameters = Namespace(DA=DA)

VirtualLab.Parameters(main_parameters,RunDA=AnalyseMLP)

VirtualLab.DA()

