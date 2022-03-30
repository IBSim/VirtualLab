#!/usr/bin/env python3
################################################################################
### HEADER
################################################################################
import sys
sys.dont_write_bytecode=True
from Scripts.Common.VirtualLab import VLSetup

################################################################################
### SETUP
################################################################################

Simulation='HIVE'
Project='.dev'
Parameters_Master='MasterSingleVoid'
#Parameters_Master='Master_HIVE'
Parameters_Var= None


################################################################################
### ENVIRONMENT
################################################################################

VirtualLab=VLSetup(
           Simulation,
           Project)

VirtualLab.Settings(
           Mode='T',
           Launcher='Process',
           NbJobs=1)

# Create directories and Parameter files for simulation
VirtualLab.Parameters(
           Parameters_Master,
           Parameters_Var,
           RunMesh=True,
           RunSim=True,
           RunDA=True)

# Creates meshes
VirtualLab.Mesh(
           ShowMesh=False,
           MeshCheck= None)

# Run Pre-Sim calculations, CodeAster and Post-Sim calculations/imaging
VirtualLab.Sim(
           RunPreAster=True, #False,
           RunAster=True, #False
           RunPostAster=True,
           ShowRes=False)

VirtualLab.DA()
