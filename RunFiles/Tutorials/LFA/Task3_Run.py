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
Parameters_Master='TrainingParameters'
Parameters_Var='Parametric_1'
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

VirtualLab.Control(
           RunMesh=False,
           RunSim=True,
           Port=None)

VirtualLab.Mesh(
           ShowMesh=False,
           MeshCheck=None)

VirtualLab.Sim(
           RunPreAster=True,
           RunAster=False,
           RunPostAster=True,
           ShowRes=False,
           ncpus=1,
           memory=2,
           mpi_nbcpu=1,
           mpi_nbnoeud=1)

VirtualLab.Cleanup()
