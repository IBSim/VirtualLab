#!/usr/bin/env python3
'''
This script demonstrates how the data collected in DataCollect.py
can be used to create 3D surrogate models of the temperature field, 
which can be used to identify inverse solutions. 
'''

import requests
import os
from types import SimpleNamespace as Namespace
from Scripts.Common.VirtualLab import VLSetup

CoilType='Pancake' 
PCA_Analysis = False
ModelType = 'GPR' # this can be GPR or MLP
CreateModel = True
InverseAnalysis = True

GUI = True

# ====================================================================
# Setup VirtualLab
VirtualLab=VLSetup('HIVE','ML_analysis')

VirtualLab.Settings(Launcher='sequential',NbJobs=1,Mode='t')

# ====================================================================
# check data has been created
DataFile = '{}_coil/TempNodal.hdf'.format(CoilType)
if not VirtualLab.InProject(DataFile):
    print("Data doesn't exist, so downloading. This may take a while")
    # download data
    DataFileFull = "{}/{}".format(VirtualLab.GetProjectDir(),DataFile)
    r = requests.get('https://zenodo.org/record/8300663/files/TempNodal.hdf')
    os.makedirs(os.path.dirname(DataFileFull),exist_ok=True)
    with open(DataFileFull,'wb') as f:
        f.write(r.content)    
    # download mesh
    r = requests.get('https://zenodo.org/record/8300663/files/HIVE_component.med')
    os.makedirs(VirtualLab.Mesh.OutputDir,exist_ok=True)
    with open("{}/HIVE_component.med".format(VirtualLab.Mesh.OutputDir),'wb') as f:
        f.write(r.content)    

# ====================================================================
# calculate reconstruction error vs the number of principal components
if PCA_Analysis:
    DA = Namespace()
    DA.Name = 'Analysis/{}/InverseSolution_T/PCA_Sensitivity'.format(CoilType)
    DA.File = ('MLtools','PCA_Sensitivity')
    DA.TrainData = [DataFile, 'Features', 'Temperature',{'group':'Train'}]
    DA.TestData = [DataFile, 'Features', 'Temperature',{'group':'Test'}]

    main_parameters = Namespace(DA=DA)

    VirtualLab.Parameters(main_parameters,RunDA=PCA_Analysis)

    VirtualLab.DA()

# ====================================================================
# Create ML model
if ModelType=='MLP' and CreateModel:
    # Create MLP model
    main_parameters = Namespace()

    ML = Namespace()
    ML.Name = 'Temperature/{}/MLP'.format(CoilType)
    ML.File = ('NN_Models','MLP_PCA_hdf5')
    ML.TrainingParameters = {'Epochs':1000,'lr':0.005}
    ML.TrainData = [DataFile, 'Features', 'Temperature',{'group':'Train'}]
    ML.ValidationData = [DataFile, 'Features', 'Temperature',{'group':'Test'}]
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
    ML.Name = 'Temperature/{}/GPR'.format(CoilType)
    ML.File = ('GPR_Models','GPR_PCA_hdf5')
    ML.TrainingParameters = {'Epochs':1000,'lr':0.05}
    ML.TrainData = [DataFile, 'Features', 'Temperature',{'group':'Train'}]
    ML.ModelParameters = {'kernel':'Matern_2.5','min_noise':1e-8,'noise_init':1e-6}
    ML.Metric = {'nb_components':20}
    main_parameters.ML = ML

    VirtualLab.Parameters(main_parameters)

    # generate GPR models
    VirtualLab.ML()

elif CreateModel:
    raise ValueError("Unknown ModelType '{}'. this must either be 'GPR' or 'MLP".format(ModelType))

# ====================================================================
# Use model to perform analysis
if InverseAnalysis:
    main_parameters = Namespace()

    DA = Namespace()
    if ModelType=='GPR':
        DA.Name = 'Analysis/{}/InverseSolution_T/GPR'.format(CoilType)
        DA.File = ('InverseSolution','AnalysisT_GPR')
        DA.MLModel = 'Temperature/{}/GPR'.format(CoilType)
    elif ModelType=='MLP':
        DA.Name = 'Analysis/{}/InverseSolution_T/MLP'.format(CoilType)
        DA.File = ('InverseSolution','AnalysisT_MLP')
        DA.MLModel = 'Temperature/{}/MLP'.format(CoilType)

    DA.MeshName = 'HIVE_component' # name of the mesh used to generate the analysis (see DataCollect.py)
    DA.TestData = [DataFile, 'Features', 'Temperature',{'group':'Test'}]
    # create comparison plots for the following indexes of the test dataset. This can be any number(s) from 0 to 299 (the size of the test dataset)
    DA.Index = [2]
    # solve inverse problem for reaching specific temperature
    DA.DesiredTemp = 600
    DA.PVGUI = GUI
    main_parameters.DA = DA

    VirtualLab.Parameters(main_parameters)

    VirtualLab.DA()
