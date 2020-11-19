#!/usr/bin/env python3
################################################################################
### HEADER
################################################################################
import sys
from os.path import dirname, abspath
sys.dont_write_bytecode=True
sys.path.append(dirname(dirname(abspath(__file__))))
from Scripts.Common.VirtualLab import VLSetup

################################################################################
### SETUP
################################################################################

Simulation='LFA'
Project='Tutorials'
StudyName='Training'
Parameters_Master='TrainingParameters_Task5'
Parameters_Var='Parametric_1_Task5'
Mode='Interactive'

################################################################################
### ENVIRONMENT
################################################################################

VirtualLab=VLSetup(
           Simulation,
           Project,
           StudyName,
           Parameters_Master,
           Parameters_Var,
           Mode)

VirtualLab.Control(RunMesh=False)

VirtualLab.Mesh()

VirtualLab.Sim(NumThreads=3,ShowRes=True)

VirtualLab.Cleanup()