#!/usr/bin/env python3
#===============================================================================
# Header
#===============================================================================

import os
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

if not os.path.isfile("{}/HIVE/Tutorials/Meshes/AMAZE.med".format(VirtualLab._OutputDir)):
    from Task2_mesh import get_mesh
    get_mesh() # create mesh which will be used for this analysis

VirtualLab.Settings(
           Mode='Interactive',
           Launcher='Process',
           NbJobs=1
           )

VirtualLab.Parameters(
           Parameters_Master,
           Parameters_Var,
           RunCT_Scan=True,
           RunCT_Recon=False
           )

#===============================================================================
# Methods
#===============================================================================

VirtualLab.CT_Scan()
