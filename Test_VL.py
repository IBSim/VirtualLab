#!/usr/bin/env python3

import os
import sys
import importlib
sys.dont_write_bytecode=True

# Include these lines to execute VL_RunFile from anywhere.
# To succeed, VL_DIR must be in PATH and VL_exe must be executable.
import shutil
VL_exe = "Test_VL.py"
try:
	from Scripts.Common import VLSetup
except:
	if shutil.which(VL_exe) == None:
		print("Can't find VirtualLab in PATH, exiting.")
		exit()
	else:
		print("VirtualLab found in PATH.")
		VL_DIR = os.path.dirname(shutil.which(VL_exe))
	sys.path.insert(0, VL_DIR)
	from Scripts.Common import VLSetup
else:
	print("Currently running from VL_DIR")
	from Scripts.Common import VLSetup

Simulation = 'Tensile'
StudyDir = 'Example'
StudyName = 'Training'
Input = {'Main' : 'Input', 'Single' : 'SingleStudy'}

# kwarg 'mode' has 3 options - interactive, continuous or headless (default)
VirtualLab = VLSetup(Simulation, StudyDir, StudyName, Input, mode = "interactive", VL_exe = VL_exe)

# Create temporary directories and files
VirtualLab.Create(RunMesh = True, RunAster=True)

# Creates meshes
VirtualLab.Mesh()

# Run Pre-Sim calculations, CodeAster and Post-Sim calculations/imaging
VirtualLab.Aster(ncpus=2, Memory=2)
VirtualLab.PostProc(ShowRes=False)

# Remove tmp folders
VirtualLab.Cleanup()
