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

Simulation='CIL'
Project='Tutorials'
Parameters_Master='TrainingParameters_CIL_Ex1'
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
           RunCT_Scan=True,
           RunCT_Recon=True,
           )

#===============================================================================
# Methods
#===============================================================================

VirtualLab.CT_Scan()
VirtualLab.CT_Recon()
