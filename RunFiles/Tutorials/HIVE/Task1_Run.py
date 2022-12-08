#!/usr/bin/env python3
#===============================================================================
# Header
import sys
sys.dont_write_bytecode=True
from Scripts.Common.VirtualLab import VLSetup

#===============================================================================
# Setup

Simulation='HIVE'
Project='Tutorials'
Parameters_Master='TrainingParameters'
Parameters_Var=None

#===============================================================================
# Environment

VirtualLab=VLSetup(
           Simulation,
           Project)

VirtualLab.Settings(
           Mode='Interactive',
           Launcher='Process',
           NbJobs=1)

VirtualLab.Parameters(
           Parameters_Master,
           Parameters_Var
           )

VirtualLab.Mesh(ShowMesh=False,
           MeshCheck=None)

VirtualLab.Sim(RunERMES=False,RunAster=False,ShowRes=True)

VirtualLab.DA()
