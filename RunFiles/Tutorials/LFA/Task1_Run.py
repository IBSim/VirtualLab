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
Parameters_Master='TrainingParameters'
Parameters_Var='Parametric_1'

#===============================================================================
# Environment
#===============================================================================

VirtualLab=VLSetup(
           Simulation,
           Project)

VirtualLab.Settings(
           Mode='Interactive',
           Launcher='Process',
           NbJobs=2)

VirtualLab.Parameters(
           Parameters_Master,
           Parameters_Var)

#===============================================================================
# Methods
#===============================================================================

VirtualLab.Mesh(ShowMesh=True)

VirtualLab.Sim(ShowRes=True)

VirtualLab.DA()
