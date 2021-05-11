#!/usr/bin/env python3

import sys
import os
sys.dont_write_bytecode=True
from Scripts.Common.VirtualLab import VLSetup

NbThreads = 5
N = int(os.environ.get('SLURM_NTASKS',NbThreads))


Simulation = 'HIVE'
Project = 'AMAZE'
StudyName = 'TestData'
Parameters_Master='devMaster'
Parameters_Var=None
Mode = "T"

VirtualLab = VLSetup(Simulation, Project, StudyName,
		     Parameters_Master, Parameters_Var,Mode)

# Create Parameter files for simulation
VirtualLab.Control(RunMesh=0, RunSim=0, RunDA=1)

# Creates meshes
VirtualLab.devMesh(NumThreads=N, launcher='MPI', ShowMesh=0, MeshCheck=None)

# Run Pre-Sim calculations, CodeAster and Post-Sim calculations/imaging
VirtualLab.devSim(NumThreads=N, launcher='MPI', RunPreAster=1, RunAster=0, RunPostAster=1, ShowRes=0, mpi_nbcpu=1, mpi_nbnoeud=1, ncpus=1, memory=10,)

# Run ML routine
VirtualLab.devDA(NumThreads=N, launcher='Process')

# Remove tmp folders
VirtualLab.Cleanup()
