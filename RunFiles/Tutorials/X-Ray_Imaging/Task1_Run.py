#!/usr/bin/env python3

#===============================================================================
# Header

import os
import sys
sys.dont_write_bytecode=True

from Scripts.Common.VirtualLab import VLSetup
from Task1_mesh import get_mesh

Simulation='Examples'
Project='Dragon'
Parameters_Master='TrainingParameters_GVXR-Draig'
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

VirtualLab.CT_Scan()