#!/usr/bin/env python3

import sys
sys.dont_write_bytecode=True
from Scripts.Common import VLSetup

Simulation = 'Tensile'
StudyDir = 'Example'
StudyName = 'Training'
Input = {'Main' : 'Input'}

# kwarg 'mode' has 3 options - interactive, continuous or headless (default)
VirtualLab = VLSetup(Simulation, StudyDir, StudyName, Input, mode = "interactive")

# Create temporary directories and files
VirtualLab.Create()

# Creates meshes
VirtualLab.Mesh()

# Run Pre-Sim calculations, CodeAster and Post-Sim calculations/imaging
VirtualLab.Aster()
VirtualLab.PostProc(ShowRes=True)

# Remove tmp folders
VirtualLab.Cleanup()

