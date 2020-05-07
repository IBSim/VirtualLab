#!/usr/bin/env python3

import os
import sys
import importlib
sys.dont_write_bytecode=True
from Scripts.Common import VLSetup

Simulation = 'Tensile'
StudyDir = 'Example'
StudyName = 'Training'
Input = {'Parameters' : 'Input', 'Single' : 'SingleStudy'}

# kwarg 'mode' has 3 options - interactive, continuous or headless (default)
VirtualLab = VLSetup(Simulation, StudyDir, StudyName, Input, mode = "headless")

# Create temporary directories and files
VirtualLab.Create(RunMesh = True, RunAster=True)

# Creates meshes
VirtualLab.Mesh()

# Run Pre-Sim calculations, CodeAster and Post-Sim calculations/imaging
VirtualLab.Aster(ncpus=2, Memory=2)
VirtualLab.PostProc(ShowRes=False)

# Remove tmp folders
VirtualLab.Cleanup()
