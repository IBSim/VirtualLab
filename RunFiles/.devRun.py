#!/usr/bin/env python3

import sys
from os.path import dirname, abspath
sys.dont_write_bytecode=True
sys.path.append(dirname(dirname(abspath(__file__))))
from Scripts.Common.VirtualLab import VLSetup

Simulation = 'HIVE'
Project = '.dev'
StudyName = 'Testing'
Parameters_Master='Master'
# Parameters_Var='Parametric_1'
# Parameters_Var='Random'
Parameters_Var=None
Mode = "T"

VirtualLab = VLSetup(Simulation,Project,StudyName,Parameters_Master,Parameters_Var,Mode)

# Create directories and Parameter files for simulation
VirtualLab.Control(RunMesh=1, RunSim=1, RunML=0)

# Creates meshes
VirtualLab.devMesh(ShowMesh=0, MeshCheck=None, NumThreads=5)

# Run Pre-Sim calculations, CodeAster and Post-Sim calculations/imaging
VirtualLab.devSim(RunPreAster=1, RunAster=0, RunPostAster=0, NumThreads=6, ShowRes=1, mpi_nbcpu=1, mpi_nbnoeud=1, ncpus=2, memory=10, launcher='Process',onall=False)

VirtualLab.devML()

# Remove tmp folders
VirtualLab.Cleanup()
