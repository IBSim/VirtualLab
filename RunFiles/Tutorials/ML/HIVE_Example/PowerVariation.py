#!/usr/bin/env python3
'''
This script demonstrates how the data collected in #SCRIPTNAME
can be used to weigh up the trade off between the power delivered to 
the sample and the uniformity of the heating profile.


'''
import requests
import os
from types import SimpleNamespace as Namespace
from Scripts.Common.VirtualLab import VLSetup


CoilType = 'Pancake' 
ModelType = 'GPR' # this can be GPR or MLP
CreateModel = True
PVAnalysis = True

# ====================================================================
# Setup VirtualLab
VirtualLab=VLSetup('HIVE','ML_analysis')

VirtualLab.Settings(Launcher='sequential',NbJobs=1,Mode='t')

# ====================================================================
# check data has been created, if not download it
DataFile = '{}_coil/PowerVariation.hdf'.format(CoilType)
if not VirtualLab.InProject(DataFile):
    DataFileFull = "{}/{}".format(VirtualLab.GetProjectDir(),DataFile)
    print("Data doesn't exist, so downloading.")
    r = requests.get('https://zenodo.org/record/8300663/files/PowerVariation.hdf')
    os.makedirs(os.path.dirname(DataFileFull),exist_ok=True)
    with open(DataFileFull,'wb') as f:
        f.write(r.content)    

# ====================================================================
# Create ML model
if ModelType=='MLP' and CreateModel:
    # Create three MLP models with different architectures and compare their performance
    main_parameters = Namespace()
    var_parameters = Namespace() 

    ML = Namespace()
    ML.File = ('NN_Models','MLP_hdf5')
    ML.TrainingParameters = {'Epochs':1000,'lr':0.05}
    ML.TrainData = [DataFile, 'Features', [['Power'],['Variation']],{'group':'Train'}]
    ML.ValidationData = [DataFile, 'Features', [['Power'],['Variation']],{'group':'Test'}] # data used to monitor for overfitting
    ML.Seed = 100 # initial weights of MLP are randomised so this ensures reproducability
    main_parameters.ML = ML

    Architectures = [[32,32],[16,32,16],[8,16,8,4]] # the hidden layers of the MLP
    ML = Namespace(Name = [],ModelParameters=[])
    for architecture in Architectures:
        ML.ModelParameters.append({'Architecture':architecture})
        arch_str = '_'.join(map(str,architecture)) # convert architecture to string and save under that name
        ML.Name.append("PV/{}/MLP/{}".format(CoilType,arch_str))
    var_parameters.ML = ML

    DA = Namespace()
    DA.Name = "Analysis/{}/PowerVariation/MLP_Compare".format(CoilType) # results will be saved to same directory as before
    DA.File = ['PowerVariation','MLP_compare']
    DA.MLModels = var_parameters.ML.Name # use the models defined earlier
    DA.TestData = [DataFile, 'Features', [['Power'],['Variation']],{'group':'Test'}] # unseen data to analyse performance
    main_parameters.DA = DA

    VirtualLab.Parameters(main_parameters,var_parameters)

    # generate MLP models
    VirtualLab.ML()
    # analyse performance of MLP model
    VirtualLab.DA()

elif ModelType=='GPR' and CreateModel:
    # Create three GPR models each with different kernels and compare their performance
    main_parameters = Namespace()
    var_parameters = Namespace() 

    # parameters used to generate model
    ML = Namespace()
    ML.File = ('GPR_Models','GPR_hdf5')
    ML.TrainingParameters = {'Epochs':1000,'lr':0.05}
    ML.TrainData = [DataFile, 'Features', [['Power'],['Variation']],{'group':'Train'}]
    main_parameters.ML = ML

    GPR_kernels = ['RBF','Matern_1.5','Matern_2.5']
    ML = Namespace(Name = [],ModelParameters=[])
    for kernel in GPR_kernels:
        ML.ModelParameters.append({'kernel':kernel})
        ML.Name.append("PV/{}/GPR/{}".format(CoilType,kernel))
    var_parameters.ML = ML

    # parameters used to compare models
    DA = Namespace()
    DA.Name = "Analysis/{}/PowerVariation/GPR_Compare".format(CoilType)
    DA.File = ['PowerVariation','GPR_compare']
    DA.MLModels = var_parameters.ML.Name # use the models defined above
    DA.TestData = [DataFile, 'Features', [['Power'],['Variation']],{'group':'Test'}] # unseen data to analyse performance
    main_parameters.DA = DA

    VirtualLab.Parameters(main_parameters,var_parameters)

    # generate GPR models
    VirtualLab.ML()
    # compare accuracy of the three models
    VirtualLab.DA()

elif CreateModel:
    raise ValueError("Unknown ModelType '{}'. this must either be 'GPR' or 'MLP".format(ModelType))

# ====================================================================
# create performance envelope of power versus variation
if PVAnalysis:
    if ModelType=='GPR':
        DA = Namespace()
        DA.Name = "Analysis/{}/PowerVariation/GPR_Analysis".format(CoilType)
        DA.File = ['PowerVariation','Insight_GPR']
        DA.MLModel = "PV/{}/GPR/Matern_2.5".format(CoilType) # chose a single model to gain insight from
        main_parameters = Namespace(DA=DA)

    elif ModelType=='MLP':
        DA = Namespace()
        DA.Name = "Analysis/{}/PowerVariation/MLP_Analysis".format(CoilType)
        DA.File = ['PowerVariation','Insight_MLP']
        DA.MLModel = "PV/{}/MLP/32_32".format(CoilType) # chose a single model to gain insight from
        main_parameters = Namespace(DA=DA)

    VirtualLab.Parameters(main_parameters)

    VirtualLab.DA()



