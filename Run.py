#!/usr/bin/env python3

import sys
sys.dont_write_bytecode=True
import os
from Scripts.Common import Setup

Simulation = 'Tensile'
StudyDir = 'Example'
StudyName = 'Test'
Input = {'Main' : 'Input', 'Parametric' : 'ParametricFile'}

# kwarg 'mode' has 3 options - interactive, continuous or headless (default)
Study = Setup(Simulation, StudyDir, StudyName, Input, mode = "interactive")

# Create temporary directories and files
Study.Create()

# Creates meshes and runs any other pre-procesing steps
Study.PreProc()

# Run simulation.
Study.Aster(ncpus=2, Memory=2, RunAster=True)

# Run post processing of results
Study.PostProc(ShowRes=False)

# Remove tmp folders
Study.Cleanup()

