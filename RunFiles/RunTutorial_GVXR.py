#!/usr/bin/env python3
import sys
sys.dont_write_bytecode=True
from Scripts.Common.VirtualLab import VLSetup

#===============================================================================
# Setup

Simulation='GVXR'
Project='Tutorials'
Parameters_Master='TrainingParameters_GVXR'
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
           Parameters_Var,
           RunGVXR=True)

VirtualLab.CT_Scan()
