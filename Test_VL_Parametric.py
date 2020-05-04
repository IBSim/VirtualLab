#!/usr/bin/env python3

import os
import sys
sys.dont_write_bytecode=True

from Scripts.Common import VLSetup

Simulation = 'Tensile'
StudyDir = 'Example'
StudyName = 'Training_AutoParam'
Input = {'Main' : 'Input', 'Parametric' : 'ParametricFile'}

# kwarg 'mode' has 3 options - interactive, continuous or headless (default)
VirtualLab = VLSetup(Simulation, StudyDir, StudyName, Input, mode = "continuous", port=2810)

# Create temporary directories and files
VirtualLab.Create()

# Creates meshes
VirtualLab.Mesh()

# Run Pre-Sim calculations, CodeAster and Post-Sim calculations/imaging
VirtualLab.Aster(ncpus=2, Memory=2)
VirtualLab.PostProc(ShowRes=False)

# Remove tmp folders
VirtualLab.Cleanup()
