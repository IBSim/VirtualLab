#!/usr/bin/env python3
#===============================================================================
# Header
#===============================================================================

import os
import sys
sys.dont_write_bytecode=True
import requests
from Scripts.Common.VirtualLab import VLSetup
import VLconfig

#===============================================================================
# Definitions
#===============================================================================

Simulation='Examples'
Project='Dragon'
Parameters_Master='TrainingParameters_Dragon'
Parameters_Var=None


#===============================================================================
# Environment
#===============================================================================

VirtualLab=VLSetup(
           Simulation,
           Project
           )

mesh_file = "{}/welsh-dragon-small.stl".format(VirtualLab.Mesh.OutputDir)
if not os.path.isfile(mesh_file):
    sys.path.insert(0,VirtualLab.PARAMETERS_DIR)
    from DragonMesh import get_mesh
    get_mesh(mesh_file)
    sys.path.pop(0)

VirtualLab.Settings(
           Mode='Interactive',
           Launcher='Process',
           NbJobs=1
           )

VirtualLab.Parameters(
           Parameters_Master,
           Parameters_Var,
           RunMesh=False,
           RunSim=False,
           RunDA=False
           )

#===============================================================================
# Methods
#===============================================================================

VirtualLab.Voxelise()
