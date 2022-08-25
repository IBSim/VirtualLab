#!/usr/bin/env python3
#===============================================================================
# Header
import sys
sys.dont_write_bytecode=True
from Scripts.Common.VirtualLab import VLSetup

'''
Same behaviout as Task3_Run, however Parameters_Master & Var are both represented
by a single file, TrainingParameters_Task3_a.
'''


#===============================================================================
# Setup

Simulation='Tensile'
Project='Tutorials'
Parameters_Master='TrainingParameters_Task3_a'
Parameters_Var=None

#===============================================================================
# Environment

VirtualLab=VLSetup(
           Simulation,
           Project)

VirtualLab.Settings(
           Mode='Interactive',
           Launcher='Process',
           NbJobs=2)

VirtualLab.Parameters(
           Parameters_Master,
           Parameters_Var
           )

VirtualLab.Mesh()

VirtualLab.Sim(ShowRes=True)

VirtualLab.DA()
