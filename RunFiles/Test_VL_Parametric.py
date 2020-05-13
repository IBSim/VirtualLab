#!/usr/bin/env python3

import sys
import os
sys.dont_write_bytecode=True

try: from Scripts.Common import VLSetup
except ModuleNotFoundError: 
	sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	from Scripts.Common import VLSetup

Simulation = 'Tensile'
StudyDir = 'Example'
StudyName = 'Training'
Input = {'Parameters' : 'TrainingParameters', 'Parametric':'Parametric_1'}

# kwarg 'mode' has 3 options - interactive, continuous or headless (default)
VirtualLab = VLSetup(Simulation, StudyDir, StudyName, Input, mode = "continuous")

# Create temporary directories and files
VirtualLab.Create(RunMesh = True, RunAster=True)

# Creates meshes
VirtualLab.Mesh()

# Run Pre-Sim calculations, CodeAster and Post-Sim calculations/imaging
VirtualLab.Aster(RunAster=True, ncpus=2, Memory=2)
VirtualLab.PostAster(ShowRes=False)

# Remove tmp folders
VirtualLab.Cleanup()
