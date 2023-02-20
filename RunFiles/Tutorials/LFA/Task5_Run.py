#!/usr/bin/env python3
#===============================================================================
# Header
#===============================================================================

import sys
sys.dont_write_bytecode=True
from Scripts.Common.VirtualLab import VLSetup

#===============================================================================
# Definitions
#===============================================================================

Simulation='LFA'
Project='Tutorials'
Parameters_Master='TrainingParameters_Task5'
Parameters_Var='Parametric_1_Task5'

#===============================================================================
# Environment
#===============================================================================

VirtualLab=VLSetup(
           Simulation,
           Project)

VirtualLab.Settings(
           Mode='Interactive',
           Launcher='Process',
           NbJobs=3)

VirtualLab.Parameters(
           Parameters_Master,
           Parameters_Var,
           RunMesh=False)

#===============================================================================
# Methods
#===============================================================================

VirtualLab.Mesh()

VirtualLab.Sim()

VirtualLab.DA()
