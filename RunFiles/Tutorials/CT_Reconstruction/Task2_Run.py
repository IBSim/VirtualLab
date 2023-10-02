#!/usr/bin/env python3
#===============================================================================
# Header
#===============================================================================

import os
import sys
sys.dont_write_bytecode=True
from Scripts.Common.VirtualLab import VLSetup
from Task2_mesh import get_mesh

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

if not os.path.isfile("{}/HIVE/Tutorials/Meshes/AMAZE_turn.med".format(VirtualLab._OutputDir)):
    get_mesh() # create mesh which will be used for this analysis

VirtualLab.Settings(
           Mode='Interactive',
           Launcher='Sequential',
           NbJobs=1
           )

VirtualLab.Parameters(
           Parameters_Master,
           Parameters_Var,
           RunCT_Scan=True,
           RunCT_Recon2D=False,
           )

#===============================================================================
# Methods
#===============================================================================

VirtualLab.CT_Scan()
VirtualLab.CT_Recon2D(Helical=True)
