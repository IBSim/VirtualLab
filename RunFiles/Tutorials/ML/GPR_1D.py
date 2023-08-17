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

Simulation='Unit'
Project='ML'
Parameters_Master='GPR_1D'
Parameters_Var=None

#===============================================================================
# Environment
#===============================================================================

VirtualLab=VLSetup(
           Simulation,
           Project
           )

VirtualLab.Settings(
           Mode='Headless',
           Launcher='Process',
           NbJobs=1
           )

VirtualLab.Parameters(
           Parameters_Master,
           Parameters_Var,
           RunMesh=True,
           RunSim=True,
           RunDA=True
           )

#===============================================================================
# Methods
#===============================================================================

VirtualLab.Mesh(
           ShowMesh=False,
           MeshCheck=None
           )

VirtualLab.Sim(
           RunPreAster=True,
           RunAster=True,
           RunPostAster=True,
           ShowRes=False
           )

VirtualLab.DA()