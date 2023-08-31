#!/usr/bin/env python3
'''
This script demonstrates how the data collected in DataCollect.py
can be used to create 3D surrogate models of the temperature and 
Von Mises stress fields, which can be used to identify inverse solutions.

This script assumes that temperature surrogate models have already been generated, see 
InverseSolution_T.py for more details on this.
'''

import requests
import os
from types import SimpleNamespace as Namespace
from Scripts.Common.VirtualLab import VLSetup


CoilType='Pancake' 
PCA_Analysis = False
ModelType = 'GPR' # this can be GPR or MLP
CreateModel = False
InverseAnalysis = True

GUI = True

# ====================================================================
# Setup VirtualLab
VirtualLab=VLSetup('HIVE','ML_analysis')

VirtualLab.Settings(Launcher='sequential',NbJobs=1,Mode='t')

# ====================================================================
# check data has been created
DataFile = '{}_coil/VMNodal.hdf'.format(CoilType)
if not VirtualLab.InProject(DataFile):
    print("Data doesn't exist, so downloading. This may take a while")
    # download data
    DataFileFull = "{}/{}".format(VirtualLab.GetProjectDir(),DataFile)
    r = requests.get('https://zenodo.org/record/8300663/files/VMNodal.hdf')
    os.makedirs(os.path.dirname(DataFileFull),exist_ok=True)
    with open(DataFileFull,'wb') as f:
        f.write(r.content)    

# ====================================================================
# calculate reconstruction error vs the number of principal components
if PCA_Analysis:
    DA = Namespace()
    DA.Name = 'Analysis/{}/InverseSolution_VM/PCA_Sensitivity'.format(CoilType)
    DA.File = ('MLtools','PCA_Sensitivity')
    DA.TrainData = [DataFile, 'Features', 'VonMises',{'group':'Train'}]
    DA.TestData = [DataFile, 'Features', 'VonMises',{'group':'Test'}]

    main_parameters = Namespace(DA=DA)

    VirtualLab.Parameters(main_parameters)

    VirtualLab.DA()

# ====================================================================
# Create ML model
if ModelType=='MLP' and CreateModel:
    # Create MLP model
    main_parameters = Namespace()

    ML = Namespace()
    ML.Name = 'VonMises/{}/MLP'.format(CoilType)
    ML.File = ('NN_Models','MLP_PCA_hdf5')
    ML.TrainingParameters = {'Epochs':1000,'lr':0.005}
    ML.TrainData = [DataFile, 'Features', 'VonMises',{'group':'Train'}]
    ML.ValidationData = [DataFile, 'Features', 'VonMises',{'group':'Test'}]
    ML.ModelParameters = {'Architecture':[8,16,8]} 
    ML.Seed = 100 # initial weights of MLP are randomised so this ensures reproducability
    ML.Metric = {'nb_components':20}
    main_parameters.ML = ML 

    VirtualLab.Parameters(main_parameters)

    # generate GPR models
    VirtualLab.ML()

elif ModelType=='GPR' and CreateModel:
    # Create GPR model
    main_parameters = Namespace()

    ML = Namespace()
    ML.Name = 'VonMises/{}/GPR'.format(CoilType)
    ML.File = ('GPR_Models','GPR_PCA_hdf5')
    ML.TrainingParameters = {'Epochs':1000,'lr':0.05}
    ML.TrainData = [DataFile, 'Features', 'VonMises',{'group':'Train'}]
    ML.ModelParameters = {'kernel':'Matern_2.5','min_noise':1e-8,'noise_init':1e-6}
    ML.Metric = {'nb_components':20}
    main_parameters = Namespace(ML=ML)

    VirtualLab.Parameters(main_parameters)

    # generate GPR models
    VirtualLab.ML()

# ====================================================================
# Use models (Von Mises and temperature) to perform analysis
if InverseAnalysis:
    main_parameters = Namespace()

    DA = Namespace()
    if ModelType=='GPR':
        DA.Name = 'Analysis/{}/InverseSolution_VM/GPR'.format(CoilType)
        DA.File = ('InverseSolution','AnalysisVM_GPR')
        DA.MLModel_T = 'Temperature/{}/GPR'.format(CoilType)
        DA.MLModel_VM = 'VonMises/{}/GPR'.format(CoilType)
    elif ModelType=='MLP':
        DA.Name = 'Analysis/{}/InverseSolution_VM/MLP'.format(CoilType)
        DA.File = ('InverseSolution','AnalysisVM_MLP')
        DA.MLModel_T = 'Temperature/{}/MLP'.format(CoilType)
        DA.MLModel_VM = 'VonMises/{}/MLP'.format(CoilType)
         
    DA.MeshName = 'HIVE_component' # name of the mesh used to generate the analysis (see DataCollect.py)
    DA.TestData = [DataFile, 'Features', 'VonMises',{'group':'Test'}]
    # create comparison plots for the following indexes of the test dataset. This can be any numbers up to 300 (the size of the test dataset)
    DA.Index = [2]
    DA.DesiredTemp = 600
    DA.PVGUI = GUI
    main_parameters.DA = DA 

    VirtualLab.Parameters(main_parameters)

    VirtualLab.DA()


