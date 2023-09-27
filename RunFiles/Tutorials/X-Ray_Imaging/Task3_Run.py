#!/usr/bin/env python3
#===============================================================================
# Header
#===============================================================================

import os
import sys
sys.dont_write_bytecode=True

from Scripts.Common.VirtualLab import VLSetup
from Task1_mesh import get_mesh

#===============================================================================
# Definitions
#===============================================================================

Simulation='GVXR'
Project='Tutorials'
Parameters_Master='TrainingParameters_GVXR-Draig_Nikon'
Parameters_Var=None

#===============================================================================
# Environment
#===============================================================================

VirtualLab=VLSetup(
           Simulation,
           Project
           )

mesh_file = "{}/Meshes/welsh-dragon-small.stl".format(VirtualLab.PARAMETERS_DIR)
if not os.path.isfile(mesh_file):
    get_mesh(mesh_file)

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
