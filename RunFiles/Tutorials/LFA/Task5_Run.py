#!/usr/bin/env python3
#===============================================================================
# Header
import sys
sys.dont_write_bytecode=True
from Scripts.Common.VirtualLab import VLSetup

#===============================================================================
# Header

Simulation='LFA'
Project='Tutorials'
Parameters_Master='TrainingParameters_Task5'
Parameters_Var='Parametric_1_Task5'

#===============================================================================
# Header

VirtualLab=VLSetup(
           Simulation,
           Project)

VirtualLab.Settings(
           Mode='Interactive',
           Launcher='Process',
           NbThreads=3)

VirtualLab.Parameters(
           Parameters_Master,
           Parameters_Var,
           RunMesh=False)

VirtualLab.Mesh()

VirtualLab.Sim()

VirtualLab.DA()

VirtualLab.Cleanup()
