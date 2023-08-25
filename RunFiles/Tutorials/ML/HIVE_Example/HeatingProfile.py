#!/usr/bin/env python3
'''
This script demonstrates how the data collected in DataCollect.py
can be used to predict the heating profile on the surface 
adjacent to the induction coil.
'''

from types import SimpleNamespace as Namespace
from Scripts.Common.VirtualLab import VLSetup

CoilType='Pancake' 
PCA_Analysis = False
ModelType = 'GPR' # this can be GPR or MLP
CreateModel = True
CreateImages = True

# ====================================================================
# Setup VirtualLab
VirtualLab=VLSetup('HIVE','ML_analysis')

VirtualLab.Settings(Launcher='sequential',NbJobs=1,Mode='t')

# check data has been created, if not download
DataFile = '{}_coil/JouleHeating.hdf'.format(CoilType)
if not VirtualLab.InProject(DataFile):
    pass # Download data from somewhere and HIVE component mesh

if PCA_Analysis:
    DA = Namespace()
    DA.Name = 'Analysis/{}/HeatingProfile/PCA_Sensitivity'.format(CoilType)
    DA.File = ('MLtools','PCA_Sensitivity')
    DA.TrainData = [DataFile, 'Features', 'SurfaceJH',{'group':'Train'}]
    DA.TestData = [DataFile, 'Features', 'SurfaceJH',{'group':'Test'}]

    main_parameters = Namespace(DA=DA)

    VirtualLab.Parameters(main_parameters)

    VirtualLab.DA()

if ModelType=='MLP' and CreateModel:
    # ====================================================================
    # Create MLP model
    # ====================================================================
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
    # ====================================================================
    # Create GPR model
    # ====================================================================    

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

if CreateImages:
    # ====================================================================
    # Create images comparing the model with the simulation
    # ====================================================================

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
    # DA.PVGUI = True

    main_parameters = Namespace(DA=DA)

    VirtualLab.Parameters(main_parameters)

    VirtualLab.DA()


