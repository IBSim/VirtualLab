#!/usr/bin/env python3

import sys
from os.path import dirname, abspath
sys.dont_write_bytecode=True
sys.path.append(dirname(dirname(abspath(__file__))))
from Scripts.Common.VirtualLab import VLSetup

Simulation = 'Tensile'
Project = '.dev'
StudyName = 'Testing'
Parameters_Master='Input'
Parameters_Var=None
Parameters_Var='Parametric_1'
Mode = "I"

VirtualLab = VLSetup(Simulation,Project,StudyName,Parameters_Master,Parameters_Var,Mode)

# Create directories and Parameter files for simulation
VirtualLab.Control(RunMesh=True, RunSim=True, Port=None)

# Creates meshes
VirtualLab.Mesh(ShowMesh=False, MeshCheck=None, NumThreads=1)

# Run Pre-Sim calculations, CodeAster and Post-Sim calculations/imaging
VirtualLab.Sim(RunPreAster=True, RunAster=True, RunPostAster=True, ShowRes=False, mpi_nbcpu=1, mpi_nbnoeud=1, ncpus=2, memory=10)

# Remove tmp folders
VirtualLab.Cleanup()
