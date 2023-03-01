#!/usr/bin/env python3
#===============================================================================
# Header
import sys
sys.dont_write_bytecode=True
from Scripts.Common.VirtualLab import VLSetup

#===============================================================================
# Setup

Simulation='Tensile'
Project='TensileBenchmark'
Parameters_Master='Parameters'
Parameters_Var= None #'Parametric'

#===============================================================================
# Environment

VirtualLab=VLSetup(
           Simulation,
           Project)

VirtualLab.Settings(
           Mode='T',
           Launcher='Process',
           ) #NbThreads=1

VirtualLab.Parameters(
           Parameters_Master,
           Parameters_Var,
           RunMesh = True,
           RunSim = True,
           RunDA = True
           )

VirtualLab.Mesh(
           ShowMesh=False,
           MeshCheck=None)

VirtualLab.Sim(
           RunPreAster=True,
           RunAster=False,
           RunPostAster=True,
           ShowRes=False)

VirtualLab.DA()

VirtualLab.Cleanup()
