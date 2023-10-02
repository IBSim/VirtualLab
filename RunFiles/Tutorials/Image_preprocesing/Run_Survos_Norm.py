#!/usr/bin/env python3
#script to run full CT scan and reconstruction with HIVE mesh
import sys
sys.dont_write_bytecode=True
from Scripts.Common.VirtualLab import VLSetup

#===============================================================================
# Setup

Simulation='GVXR'
Project='Norm'
Parameters_Master='TrainingParameters_preproc'
Parameters_Var=None
#===============================================================================
# Environment

VirtualLab=VLSetup(
           Simulation,
           Project)

VirtualLab.Settings(
         #   Mode='Interactive',
           Mode='Headless',
           Launcher='Process',
           NbJobs=1)

VirtualLab.Parameters(
           Parameters_Master,
           Parameters_Var,
           RunImPreProc=True)

VirtualLab.ImPreProc(
  Normalise=True,
  Register=False)