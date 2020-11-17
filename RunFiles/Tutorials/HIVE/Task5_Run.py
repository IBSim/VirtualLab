#!/usr/bin/env python3
################################################################################
### HEADER
################################################################################
import sys
from os.path import dirname, abspath, isdir
sys.dont_write_bytecode=True
sys.path.append(dirname(dirname(abspath(__file__))))
from Scripts.Common.VirtualLab import VLSetup

################################################################################
### SETUP
################################################################################

Simulation='HIVE'
Project='Tutorials'
StudyName='Training'
Parameters_Master='TrainingParameters_Task5'
Parameters_Var=None
Mode='Interactive'

## Copy files created in task 4
import shutil
import VLconfig
import os
ResDir="{}/{}/{}/{}".format(VLconfig.OutputDir,Simulation,Project,StudyName)
if isdir("{}/Sim_ERMESx2".format(ResDir)): shutil.rmtree("{}/Sim_ERMESx2".format(ResDir))
shutil.copytree("{}/Sim_ERMES".format(ResDir),"{}/Sim_ERMESx2".format(ResDir))

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
           RunSim=True)

VirtualLab.Mesh(
	   NumThreads=1,
           ShowMesh=False,
           MeshCheck=None)

VirtualLab.Sim(
	   NumThreads=1,
           RunPreAster=True,
           RunAster=True,
           RunPostAster=True,
           ShowRes=True,
           ncpus=1,
           memory=2,
           mpi_nbcpu=1,
           mpi_nbnoeud=1)

VirtualLab.Cleanup()
