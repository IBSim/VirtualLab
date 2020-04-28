#!/usr/bin/env python3

import os
import sys
import importlib
sys.dont_write_bytecode=True
from Scripts.Common import Setup

STUDY_DIR = 'Training'
SIMULATION = 'LFA'
STUDY_NAME = 'Example'
INPUTS = {'Main' : 'Input', 'Parametric' : 'ParametricFile'}

# kwarg 'mode' has 3 options - interactive, continuous or headless (default)
Study = Setup(STUDY_DIR, SIMULATION, STUDY_NAME, INPUTS, mode = "interactive", port=2810)

# Create temporary directories and files
Study.Create()

# Creates meshes and runs any other pre-procesing steps
Study.PreProc()

# Run simulation.
Study.Aster(ncpus=2, Memory=2)

# Run post processing of results
Study.PostProc(ShowRes=False)

# Remove tmp folders
Study.Cleanup()

