#!/usr/bin/env python3
#===============================================================================
# Header
#===============================================================================

import os
import sys
sys.dont_write_bytecode=True
from Scripts.Common.VirtualLab import VLSetup
from Task3_mesh import get_mesh

#===============================================================================
# Definitions
#===============================================================================

Simulation='HIVE'
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

if not os.path.isfile("{}/AMAZE.med".format(VirtualLab.Mesh.OutputDir)):
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
