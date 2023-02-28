#!/usr/bin/env python3
#===============================================================================
# Header
#===============================================================================

import sys
sys.dont_write_bytecode=True
from Scripts.Common.VirtualLab import VLSetup

'''
Same behaviour as Task3_Run, however Parameters_Master & Var are both imported,
so that changes can be made to their values in this file on the fly.
'''

#===============================================================================
# Definitions
#===============================================================================

Simulation='Tensile'
Project='Tutorials'
Parameters_Master='TrainingParameters'
Parameters_Var='Parametric_1'

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

# Import Parameters_Master and Var files and make changes here
# e.g. Master.Mesh.Thickness = 0.005
Master = VirtualLab.ImportParameters(Parameters_Master)
Var = VirtualLab.ImportParameters(Parameters_Var)

VirtualLab.Parameters(
           Master,
           Var,
           RunMesh=True,
           RunSim=True,
           RunDA=True
           )

#===============================================================================
# Methods
#===============================================================================

VirtualLab.Mesh(
           ShowMesh=False,
           MeshCheck=None
           )

VirtualLab.Sim(
           RunPreAster=True,
           RunAster=True,
           RunPostAster=True,
           ShowRes=True
           )

VirtualLab.DA()
