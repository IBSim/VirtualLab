#!/usr/bin/env python3
#===============================================================================
# Header
import sys
sys.dont_write_bytecode=True
from Scripts.Common.VirtualLab import VLSetup

'''
You will need to download the image-based dog bone sample from the following link
https://ibsim.co.uk/VirtualLab/downloads/Tensile_IBSim.med
This will need to be saved to Output/Tensile/Tutorials/Meshes
'''

#===============================================================================
# Setup

Simulation='Tensile'
Project='Tutorials'
Parameters_Master='TrainingParameters_IBSim'
Parameters_Var=None

#===============================================================================
# Environment

VirtualLab=VLSetup(
           Simulation,
           Project)

VirtualLab.Settings(
           Mode='Interactive',
           Launcher='Process',
           NbThreads=1)

VirtualLab.Parameters(
           Parameters_Master,
           Parameters_Var)

VirtualLab.Mesh()

VirtualLab.Sim(ShowRes=True)

VirtualLab.DA()

VirtualLab.Cleanup()
