#!/usr/bin/env python3
'''
This script demonstrates how the temperature field surrogate 
model generated in InverseSolution.py can be used to predict
the temperature field throughout the component from a handful of 
surface thermocouple measurements. 
'''

import sys
sys.dont_write_bytecode=True
from types import SimpleNamespace as Namespace
import numpy as np
from Scripts.Common.VirtualLab import VLSetup

CoilType='Pancake' # this can be 'Pancake' or 'HIVE'
EstimatedField_GPR = False
Sensitivity_GPR = False
Optimise_GPR = False
EstimatedField_MLP = False
Sensitivity_MLP = False
Optimise_MLP = True

# ====================================================================
# Setup VirtualLab
VirtualLab=VLSetup('HIVE','ML_analysis')

VirtualLab.Settings(Launcher='sequential',NbJobs=1)

DataFile = '{}_coil/TempNodal.hdf'.format(CoilType) # data already downloaded for previous analysis

# ====================================================================
# GPR analysis
# ====================================================================

# identify the full temperature field from the thermocouples specified by ThermocoupleConfig
# and create plots comparing this with the simulation for those specified by Index
DA = Namespace()
DA.Name = 'Analysis/{}/Thermocouple/GPR/Comparison'.format(CoilType)
DA.File = ('Thermocouple','FullFieldEstimate_GPR')
DA.MLModel = 'Temperature/{}/GPR'.format(CoilType)
DA.MeshName = 'HIVE_component' # name of the mesh used to generate the analysis (see DataCollect.py)
DA.TestData = [DataFile, 'Features', 'Temperature',{'group':'Test'}]
# create comparison plots for the following indexes of the test dataset. This can be any numbers up to 300 (the size of the test dataset)
DA.Index = [1]
DA.ThermocoupleConfig = [['TileSideA',0.5,0.5], 
                        ['TileFront',0.5,0.5], 
                        ['TileSideB',0.5,0.5], 
                        ['TileBack',0.5,0.5],
                        ['BlockFront',0.5,0.5], 
                        ['BlockBack',0.5,0.5], 
                        ['BlockBottom',0.5,0.5]]

main_parameters = Namespace(DA=DA)

VirtualLab.Parameters(main_parameters,RunDA=EstimatedField_GPR)

VirtualLab.DA()


# highligh the sensitivity of the results to the placement of the thermocouples
DA = Namespace()
DA.Name = 'Analysis/{}/Thermocouple/GPR/Sensitivity'.format(CoilType)
DA.File = ('Thermocouple','Sensitivity_GPR')
DA.MLModel = 'Temperature/{}/GPR'.format(CoilType)
DA.MeshName = 'HIVE_component' # name of the mesh used to generate the analysis (see DataCollect.py)
# create comparison plots for the following indexes of the test dataset. This can be any numbers up to 300 (the size of the test dataset)
DA.TestData = [DataFile, 'Features', 'Temperature',{'group':'Test'}]
DA.CandidateSurfaces = ['TileSideA','TileSideB','TileFront','TileBack','BlockFront','BlockBack','BlockBottom']
DA.NbThermocouples = 3
DA.NbConfig = 5 # number of random combinations of thermocouple placements to test

main_parameters = Namespace(DA=DA)

VirtualLab.Parameters(main_parameters,RunDA=Sensitivity_GPR)

VirtualLab.DA()

# optimise the location of thermocouples
DA = Namespace()
DA.Name = 'Analysis/{}/Thermocouple/GPR/Optimise'.format(CoilType)
DA.File = ('Thermocouple','Optimise_GPR')
DA.MLModel = 'Temperature/{}/GPR'.format(CoilType)
DA.MeshName = 'HIVE_component' # name of the mesh used to generate the analysis (see DataCollect.py)
# create comparison plots for the following indexes of the test dataset. This can be any numbers up to 300 (the size of the test dataset)
DA.TestData = [DataFile, 'Features', 'Temperature',{'group':'Test'}]
DA.CandidateSurfaces = ['TileSideA','TileSideB','TileFront','TileBack','BlockFront','BlockBack','BlockBottom']
DA.NbThermocouples = 3

main_parameters = Namespace(DA=DA)

VirtualLab.Parameters(main_parameters,RunDA=Optimise_GPR)

VirtualLab.DA()


# ====================================================================
# MLP analysis
# ====================================================================

# identify the full temperature field from the thermocouples specified by ThermocoupleConfig
# and create plots comparing this with the simulation
DA = Namespace()
DA.Name = 'Analysis/{}/Thermocouple/MLP/Compare'.format(CoilType)
DA.File = ('Thermocouple','FullFieldEstimate_MLP')
DA.MLModel = 'Temperature/{}/MLP'.format(CoilType)
DA.MeshName = 'HIVE_component' # name of the mesh used to generate the analysis (see DataCollect.py)
DA.TestData = [DataFile, 'Features', 'Temperature',{'group':'Test'}]
# create comparison plots for the following indexes of the test dataset. This can be any numbers up to 300 (the size of the test dataset)
DA.Index = [1]
DA.ThermocoupleConfig = [['TileSideA',0.5,0.5], 
                        ['TileFront',0.5,0.5], 
                        ['TileSideB',0.5,0.5], 
                        ['TileBack',0.5,0.5],
                        ['BlockFront',0.5,0.5], 
                        ['BlockBack',0.5,0.5], 
                        ['BlockBottom',0.5,0.5]]

main_parameters = Namespace(DA=DA)

VirtualLab.Parameters(main_parameters,RunDA=EstimatedField_MLP)

VirtualLab.DA()

# highligh the sensitivity of the results to the placement of the thermocouples
DA = Namespace()
DA.Name = 'Analysis/{}/Thermocouple/MLP/Sensitivity'.format(CoilType)
DA.File = ('Thermocouple','Sensitivity_MLP')
DA.MLModel = 'Temperature/{}/MLP'.format(CoilType)
DA.MeshName = 'HIVE_component' # name of the mesh used to generate the analysis (see DataCollect.py)
# create comparison plots for the following indexes of the test dataset. This can be any numbers up to 300 (the size of the test dataset)
DA.TestData = [DataFile, 'Features', 'Temperature',{'group':'Test'}]
DA.CandidateSurfaces = ['TileSideA','TileSideB','TileFront','TileBack','BlockFront','BlockBack','BlockBottom']
DA.NbThermocouples = 3
DA.NbConfig = 5 # number of random combinations of thermocouple placements to test

main_parameters = Namespace(DA=DA)

VirtualLab.Parameters(main_parameters,RunDA=Sensitivity_MLP)

VirtualLab.DA()

# optimise the location of thermocouples
DA = Namespace()
DA.Name = 'Analysis/{}/Thermocouple/MLP/Optimise'.format(CoilType)
DA.File = ('Thermocouple','Optimise_MLP')
DA.MLModel = 'Temperature/{}/MLP'.format(CoilType)
DA.MeshName = 'HIVE_component' # name of the mesh used to generate the analysis (see DataCollect.py)
# create comparison plots for the following indexes of the test dataset. This can be any numbers up to 300 (the size of the test dataset)
DA.TestData = [DataFile, 'Features', 'Temperature',{'group':'Test'}]
DA.CandidateSurfaces = ['TileSideA','TileSideB','TileFront','TileBack','BlockFront','BlockBack','BlockBottom']
DA.NbThermocouples = 3

main_parameters = Namespace(DA=DA)

VirtualLab.Parameters(main_parameters,RunDA=Optimise_MLP)

VirtualLab.DA()