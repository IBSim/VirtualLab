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
Parameters_Master='TrainingParameters_CIL_Ex2'
Parameters_Var='TrainingParameters_CIL_var'

#===============================================================================
# Environment
#===============================================================================

VirtualLab=VLSetup(
           Simulation,
           Project
           )

VirtualLab.Settings(
           Mode='Interactive',
           Launcher='Sequential',
           NbJobs=1
           )

VirtualLab.Parameters(
           Parameters_Master,
           Parameters_Var,
           RunMesh=False,
           RunCT_Scan=True,
           RunCT_Recon2D=False,
           )

#===============================================================================
# Methods
#===============================================================================
VirtualLab.Mesh(ShowMesh=False)
VirtualLab.CT_Scan()
VirtualLab.CT_Recon2D()
