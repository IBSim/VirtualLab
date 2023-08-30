#!/usr/bin/env python3
'''
This script demonstrates how the data collected in DataCollect.py
can be used to predict the heating profile on the surface 
adjacent to the induction coil.
'''
import requests
import os
from types import SimpleNamespace as Namespace
from Scripts.Common.VirtualLab import VLSetup

CoilType='Pancake' 
PCA_Analysis = False
ModelType = 'GPR' # this can be GPR or MLP
CreateModel = True
CreateImages = True

GUI = True

# ====================================================================
# Setup VirtualLab
VirtualLab=VLSetup('HIVE','ML_analysis')

VirtualLab.Settings(Launcher='sequential',NbJobs=1,Mode='t')

# ====================================================================
# check data has been created
DataFile = '{}_coil/JouleHeating.hdf'.format(CoilType)
if not VirtualLab.InProject(DataFile):
    print("Data doesn't exist, so downloading. This may take a while")
    # download data
    DataFileFull = "{}/{}".format(VirtualLab.GetProjectDir(),DataFile)
    r = requests.get('https://zenodo.org/record/8300663/files/JouleHeating.hdf')
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
    DA.Name = 'Analysis/{}/HeatingProfile/PCA_Sensitivity'.format(CoilType)
    DA.File = ('MLtools','PCA_Sensitivity')
    DA.TrainData = [DataFile, 'Features', 'SurfaceJH',{'group':'Train'}]
    DA.TestData = [DataFile, 'Features', 'SurfaceJH',{'group':'Test'}]

    main_parameters = Namespace(DA=DA)

    VirtualLab.Parameters(main_parameters)

    VirtualLab.DA()

# ====================================================================
# Create ML model
if ModelType=='MLP' and CreateModel:
    # Create MLP model
    main_parameters = Namespace()
    
    ML = Namespace()
    ML.Name = 'HeatProfile/{}/MLP'.format(CoilType)
    ML.File = ('NN_Models','MLP_PCA_hdf5')
    ML.TrainingParameters = {'Epochs':1000,'lr':0.005}
    ML.TrainData = [DataFile, 'Features', 'SurfaceJH',{'group':'Train'}]
    ML.ValidationData = [DataFile, 'Features', 'SurfaceJH',{'group':'Test'}]
    ML.ModelParameters = {'Architecture':[8,16,8]} 
    ML.Seed = 100 # initial weights of MLP are randomised so this ensures reproducability
    ML.Metric = {'threshold':0.99}
    main_parameters.ML = ML

    VirtualLab.Parameters(main_parameters)

    # generate GPR models
    VirtualLab.ML()

elif ModelType=='GPR' and CreateModel:
    # Create GPR model
    main_parameters = Namespace()

    ML = Namespace()
    ML.Name = 'HeatProfile/{}/GPR'.format(CoilType)
    ML.File = ('GPR_Models','GPR_PCA_hdf5')
    ML.TrainingParameters = {'Epochs':1000,'lr':0.05}
    ML.TrainData = [DataFile, 'Features', 'SurfaceJH',{'group':'Train'}]
    ML.ModelParameters = {'kernel':'Matern_2.5','min_noise':1e-8,'noise_init':1e-6}
    ML.Metric = {'threshold':0.99}
    main_parameters.ML = ML  

    VirtualLab.Parameters(main_parameters)

    # generate GPR models
    VirtualLab.ML()

elif CreateModel:
    raise ValueError("Unknown ModelType '{}'. this must either be 'GPR' or 'MLP".format(ModelType))

# ====================================================================
# Create images comparing the model with the simulation
if CreateImages:

    main_parameters = Namespace()
    DA = Namespace()

    if ModelType=='GPR':
        DA.Name = 'Analysis/{}/HeatingProfile/GPR'.format(CoilType)
        DA.File = ('HeatingProfile','CreateImage_GPR')
        DA.MLModel = 'HeatProfile/{}/GPR'.format(CoilType)
    elif ModelType=='MLP':
        DA.Name = 'Analysis/{}/HeatingProfile/MLP'.format(CoilType)
        DA.File = ('HeatingProfile','CreateImage_MLP')
        DA.MLModel = 'HeatProfile/{}/MLP'.format(CoilType)

    DA.MeshName = 'HIVE_component' # name of the mesh used to generate the analysis (see DataCollect.py)
    DA.TestData = [DataFile, 'Features', 'SurfaceJH',{'group':'Test'}]
    # create comparison plots for the following indexes of the test dataset. This can be any numbers up to 300 (the size of the test dataset)
    DA.Index = [1]
    DA.PVGUI = GUI

    main_parameters = Namespace(DA=DA)

    VirtualLab.Parameters(main_parameters)

    VirtualLab.DA()


