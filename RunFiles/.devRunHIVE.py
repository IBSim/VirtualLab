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

Simulation='HIVE'
Project='.dev'
StudyName='SingleVoidTests'
Parameters_Master='MasterSingleVoid'
#Parameters_Master='Master_HIVE'
Parameters_Var= None
Mode='T'

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

# Create directories and Parameter files for simulation
VirtualLab.Control(
           RunMesh=True, #True
           RunSim=True,
           RunML=0)

# Creates meshes
VirtualLab.devMesh(
           ShowMesh=False,
           MeshCheck= None, #'Mesh_SingleVoid', None,
           NumThreads=2) 

# Run Pre-Sim calculations, CodeAster and Post-Sim calculations/imaging
VirtualLab.devSim(
           RunPreAster=True, #False,
           RunAster=True, #False
           RunPostAster=True,
           NumThreads=2,
           ShowRes=True, #False,
           mpi_nbcpu=1,
           mpi_nbnoeud=1,
           ncpus=2,
           memory=6,
           launcher='Process',
           onall=False)

VirtualLab.devML()

# Remove tmp folders
VirtualLab.Cleanup()
