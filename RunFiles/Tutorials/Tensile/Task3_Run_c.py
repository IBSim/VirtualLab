#!/usr/bin/env python3
#===============================================================================
# Header
#===============================================================================

import sys
sys.dont_write_bytecode=True
from Scripts.Common.VirtualLab import VLSetup
from types import SimpleNamespace as Namespace

'''
Same behaviout as Task3_Run, however Parameters_Var is created in this file.
'''

#===============================================================================
# Definitions
#===============================================================================

Simulation='Tensile'
Project='Tutorials'
Parameters_Master='TrainingParameters'
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
           NbJobs=2
           )

# Create namespace containing varying mesh parameters
Mesh = Namespace(
       Name = ['Notch2','Notch3'],
       Rad_a = [0.001,0.002],
       Rad_b = [0.001,0.0005]
       )

# Create namespace containing varying sim parameters
Sim = Namespace(
      Name = ['ParametricSim1', 'ParametricSim2'],
      Mesh = ['Notch2', 'Notch3']
      )

# Attach Mesh and Sim to Var namespace (so it behaves like a module)
Var = Namespace(
      Mesh=Mesh,
      Sim=Sim
      )

VirtualLab.Parameters(
           Parameters_Master,
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
