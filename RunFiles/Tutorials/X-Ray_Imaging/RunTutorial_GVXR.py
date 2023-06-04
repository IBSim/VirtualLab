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
           RunMesh=True,
           RunCT_Scan=True
           )

#===============================================================================
# Methods
#===============================================================================
VirtualLab.Mesh(ShowMesh=False)
VirtualLab.CT_Scan()
