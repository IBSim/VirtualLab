#!/usr/bin/env python3

import sys
sys.dont_write_bytecode=True
import os
from Scripts.Common import Setup

Simulation = 'Tensile'
StudyDir = 'Testing'
StudyName = 'NewParametric'
Input = {'Main' : 'Input', 'Single' : 'EM_load'}

# kwarg 'mode' has 3 options - interactive, continuous or headless (default)
Study = Setup(Simulation, StudyDir, StudyName, Input, mode = "interactive", port=2810)

# Create temporary directories and files
Study.Create()

# Creates meshes and runs any other pre-procesing steps
Study.PreProc()

# Run simulation.
Study.Aster(ncpus=2, Memory=2, RunAster=False)

# Run post processing of results
Study.PostProc(ShowRes=False)

# Remove tmp folders
#Study.Cleanup()

