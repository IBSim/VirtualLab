#!/usr/bin/env python3

import sys
from os.path import dirname, abspath
sys.dont_write_bytecode=True
sys.path.append(dirname(dirname(abspath(__file__))))
from Scripts.Common.VirtualLab import VLSetup

Simulation = 'LFA'
Project = '.dev'
StudyName = 'Testing'
Parameters_Master='Master'
Parameters_Var=None
Mode = "T"

VirtualLab = VLSetup(Simulation,Project,StudyName,Parameters_Master,Parameters_Var,Mode)

# Create directories and Parameter files for simulation
VirtualLab.Control(RunMesh=1, RunSim=1, RunML=1)

# Creates meshes
VirtualLab.Mesh(ShowMesh=False, MeshCheck=None, NumThreads=4)

# Run Pre-Sim calculations, CodeAster and Post-Sim calculations/imaging
VirtualLab.Sim(RunPreAster=True, RunAster=1, RunPostAster=True, NumThreads=4, ShowRes=False, mpi_nbcpu=1, mpi_nbnoeud=1, ncpus=2, memory=10)

VirtualLab.ML()

# Remove tmp folders
VirtualLab.Cleanup()
