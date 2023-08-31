#!/usr/bin/env python3
'''
This script demonstrates how the temperature field surrogate 
model generated in InverseSolution.py can be used to predict
the temperature field throughout the component from a handful of 
surface thermocouple measurements. 
'''

from types import SimpleNamespace as Namespace
from Scripts.Common.VirtualLab import VLSetup

CoilType='Pancake' 
ModelType = 'MLP' # this can be GPR or MLP
EstimateField = True
Sensitivity = False
Optimise = False

GUI = False

# ====================================================================
# Setup VirtualLab
VirtualLab=VLSetup('HIVE','ML_analysis')

VirtualLab.Settings(Launcher='sequential',NbJobs=1,Mode='t')

DataFile = '{}_coil/TempNodal.hdf'.format(CoilType) # data already downloaded for previous analysis

# ====================================================================
# Identify the full temperature field using only the thermocouples data
# and create plots comparing this with the simulation 
if EstimateField:
    main_parameters = Namespace()

    DA = Namespace()
    DA.Name = 'Analysis/{}/Thermocouple/{}/EstimateField'.format(CoilType,ModelType)
    DA.File = ('Thermocouple','FullFieldEstimate_{}'.format(ModelType))
    DA.MLModel = 'Temperature/{}/{}'.format(CoilType,ModelType)
    DA.MeshName = 'HIVE_component' # name of the mesh used to generate the analysis (see DataCollect.py)
    DA.TestData = [DataFile, 'Features', 'Temperature',{'group':'Test'}]
    # create comparison plots for the following indexes of the test dataset. This can be any numbers up to 300 (the size of the test dataset)
    DA.Index = [7] 
    # Location of thermocouples
    DA.ThermocoupleConfig = [['TileSideA',0.5,0.5], 
                            ['TileFront',0.5,0.5], 
                            ['TileSideB',0.5,0.5], 
                            ['TileBack',0.5,0.5],
                            ['BlockFront',0.5,0.5], 
                            ['BlockBack',0.5,0.5], 
                            ['BlockBottom',0.5,0.5]]
    DA.PVGUI = GUI
    main_parameters.DA = DA

    VirtualLab.Parameters(main_parameters)

    VirtualLab.DA()

# ====================================================================
# Show the sensitivity of the results to the placement of the thermocouples
if Sensitivity:
    main_parameters = Namespace()

    DA = Namespace()
    DA.Name = 'Analysis/{}/Thermocouple/{}/Sensitivity'.format(CoilType,ModelType)
    DA.File = ('Thermocouple','Sensitivity_{}'.format(ModelType))
    DA.MLModel = 'Temperature/{}/{}'.format(CoilType,ModelType)
    DA.MeshName = 'HIVE_component' # name of the mesh used to generate the analysis (see DataCollect.py)
    DA.TestData = [DataFile, 'Features', 'Temperature',{'group':'Test'}]
    DA.CandidateSurfaces = ['TileSideA','TileSideB','TileFront','TileBack','BlockFront','BlockBack','BlockBottom']
    DA.NbThermocouples = 4
    DA.NbConfig = 5 # number of random combinations of thermocouple placements to test
    DA.PVGUI = GUI
    main_parameters.DA = DA

    VirtualLab.Parameters(main_parameters)

    VirtualLab.DA()

# ====================================================================
# Optimise the location of the thermocouples
if Optimise:
    NbThermocouple = 3

    main_parameters = Namespace()

    DA = Namespace()
    DA.Name = 'Analysis/{}/Thermocouple/{}/Optimise_{}'.format(CoilType,ModelType,NbThermocouple)
    DA.File = ('Thermocouple','Optimise_{}'.format(ModelType))
    DA.MLModel = 'Temperature/{}/{}'.format(CoilType,ModelType)
    DA.MeshName = 'HIVE_component' # name of the mesh used to generate the analysis (see DataCollect.py)
    DA.TestData = [DataFile, 'Features', 'Temperature',{'group':'Test'}]
    DA.CandidateSurfaces = ['TileSideA','TileSideB','TileFront','TileBack','BlockFront','BlockBack','BlockBottom']
    DA.NbThermocouples = NbThermocouple
    DA.GeneticAlgorithm = {'NbGen':5,'NbPop':20,'NbExample':5,'seed':100}
    DA.PVGUI = GUI
    main_parameters.DA = DA

    VirtualLab.Parameters(main_parameters)

    VirtualLab.DA()
