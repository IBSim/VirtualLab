#!/usr/bin/env python3

import os
import sys
import importlib
sys.dont_write_bytecode=True
from Scripts.Common import Setup

STUDY_DIR = 'Training'
SIMULATION = 'Tensile'
STUDY_NAME = 'Example'
INPUTS = {'Main' : 'Input', 'Single' : 'SingleStudy'}

# kwarg 'mode' has 3 options - interactive, continuous or headless (default)
Study = Setup(STUDY_DIR, SIMULATION, STUDY_NAME, INPUTS, mode = "headless")

# Create temporary directories and files
Study.Create()

# Create meshes.
Study.Mesh()

# Run simulation.
Study.Aster()

# Run post processing of results
Study.PostProc()

# Remove tmp folders
Study.Cleanup()