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

Simulation='GVXR'
Project='Tutorials'
Parameters_Master='TrainingParameters_GVXR'
Parameters_Var='TrainingParameters_GVXR_var'

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
           NbJobs=2
           )

VirtualLab.Parameters(
           Parameters_Master,
           Parameters_Var,
           RunCT_Scan=True
           )

#===============================================================================
# Methods
#===============================================================================

VirtualLab.CT_Scan()
