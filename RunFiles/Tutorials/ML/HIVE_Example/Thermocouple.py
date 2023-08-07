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
AnalyseGPR = True

# ====================================================================
# Setup VirtualLab
VirtualLab=VLSetup('HIVE','ML_analysis')

VirtualLab.Settings(Launcher='sequential',NbJobs=1)

# check data has been created, if not download
DataFile = '{}_coil/TempNodal.hdf'.format(CoilType)
if not VirtualLab.InProject(DataFile):
    pass # Download data from somewhere (and possibly mesh)

# ====================================================================
# GPR analysis
# ====================================================================

DA = Namespace()
DA.Name = 'Analysis/Thermocouple/GPR_{}'.format(CoilType)
DA.File = ('Thermocouple','FullFieldEstimate')
DA.MLModel = 'Temperature/GPR'
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
                        ['BlockBottom',0.5,0.5]][:5]

main_parameters = Namespace(DA=DA)

VirtualLab.Parameters(main_parameters,RunDA=AnalyseGPR)

VirtualLab.DA()