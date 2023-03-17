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

Simulation='HIVE'
Project='Tutorials'
Parameters_Master='TurnParameters'
Parameters_Var=None

#===============================================================================
# Environment
#===============================================================================

VirtualLab=VLSetup(
           Simulation,
           Project
           )

VirtualLab.Settings(
           Mode='Interactive',
           Launcher='Process',
           NbJobs=1
           )

VirtualLab.Parameters(
           Parameters_Master,
           Parameters_Var,
           RunSim=False,
           RunMesh=True
           )

#===============================================================================
# Methods
#===============================================================================

VirtualLab.Mesh(
           ShowMesh=True,
           MeshCheck=None
           )

VirtualLab.Sim(
           RunERMES=False,
           RunAster=False,
           ShowRes=False
           )

VirtualLab.DA()
