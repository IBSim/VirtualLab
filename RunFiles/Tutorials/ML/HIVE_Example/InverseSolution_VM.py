#!/usr/bin/env python3
'''
This script demonstrates how the data collected in DataCollect.py
can be used to create 3D surrogate models of the temperature and 
Von Mises stress fields, which can be used to identify inverse solutions.

This script assumes that temperature surrogate models have already been generated, see 
InverseSolution_T.py for more details on this.
'''

from types import SimpleNamespace as Namespace
from Scripts.Common.VirtualLab import VLSetup


CoilType='Pancake' 
PCA_Analysis = False
ModelType = 'GPR' # this can be GPR or MLP
CreateModel = True
InverseAnalysis = True

# ====================================================================
# Setup VirtualLab
VirtualLab=VLSetup('HIVE','ML_analysis')

VirtualLab.Settings(Launcher='sequential',NbJobs=1)

# check data has been created, if not download
DataFile = '{}_coil/VMNodal.hdf'.format(CoilType)
if not VirtualLab.InProject(DataFile):
    pass # Download data from somewhere (and possibly mesh)

if PCA_Analysis:
    DA = Namespace()
    DA.Name = 'Analysis/{}/InverseSolution_VM/PCA_Sensitivity'.format(CoilType)
    DA.File = ('MLtools','PCA_Sensitivity')
    DA.TrainData = [DataFile, 'Features', 'VonMises',{'group':'Train'}]
    DA.TestData = [DataFile, 'Features', 'VonMises',{'group':'Test'}]

    main_parameters = Namespace(DA=DA)

    VirtualLab.Parameters(main_parameters)

    VirtualLab.DA()

if ModelType=='MLP' and CreateModel:
    # ====================================================================
    # Create MLP model
    # ====================================================================
    main_parameters = Namespace()

    ML = Namespace()
    ML.Name = 'VonMises/{}/MLP'.format(CoilType)
    ML.File = ('NN_Models','MLP_PCA_hdf5')
    ML.TrainingParameters = {'Epochs':1000,'lr':0.005,'Print':50}
    ML.TrainData = [DataFile, 'Features', 'VonMises',{'group':'Train'}]
    ML.ValidationData = [DataFile, 'Features', 'VonMises',{'group':'Test'}]
    ML.ModelParameters = {'Architecture':[8,16,8]} 
    ML.Seed = 100 # initial weights of MLP are randomised so this ensures reproducability
    ML.Metric = {'nb_components':30}
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
    ML.Name = 'VonMises/GPR'
    ML.File = ('GPR_Models','GPR_PCA_hdf5')
    ML.TrainingParameters = {'Epochs':1000,'lr':0.05,'Print':50}
    ML.TrainData = [DataFile, 'Features', 'VonMises',{'group':'Train'}]
    ML.ModelParameters = {'kernel':'Matern_2.5'} #{'min_noise':1e-8,'noise_init':1e-6}
    ML.Metric = {'nb_components':30}

    main_parameters = Namespace(ML=ML)

    VirtualLab.Parameters(main_parameters)

    # generate GPR models
    VirtualLab.ML()


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

    main_parameters.DA = DA 

    VirtualLab.Parameters(main_parameters)

    VirtualLab.DA()


