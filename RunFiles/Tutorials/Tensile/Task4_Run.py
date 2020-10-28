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

Simulation='Tensile'
Project='Tutorials'
StudyName='Training'
Parameters_Master='TrainingParameters_Task4'
Parameters_Var='Parametric_1_Task4'
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

VirtualLab.Mesh(NumThreads=2)

VirtualLab.Sim(NumThreads=2, ShowRes=True)

VirtualLab.Cleanup()
