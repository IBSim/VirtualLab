#!/usr/bin/env python3

import sys
from os.path import dirname, abspath
sys.dont_write_bytecode=True
sys.path.append(dirname(dirname(abspath(__file__))))
from Scripts.Common import VLSetup

#try: from Scripts.Common import VLSetup
#except ModuleNotFoundError: 
#	sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#	from Scripts.Common import VLSetup		

Simulation = 'Tensile'
StudyDir = 'Example'
StudyName = 'Training'
Input = {'Parameters' : 'TrainingParameters', 'Parametric':'Parametric_1'}
Input = {'Parameters' : 'TrainingParameters'}

# kwarg 'mode' has 3 options - interactive, continuous or headless (default)
VirtualLab = VLSetup(Simulation, StudyDir, StudyName, Input, mode = "interactive")

# Create temporary directories and files
VirtualLab.Create(RunSim=False, RunMesh=True)

# Creates meshes
VirtualLab.Mesh()

# Run Pre-Sim calculations, CodeAster and Post-Sim calculations/imaging
VirtualLab.Aster(RunAster=True)
VirtualLab.PostAster(ShowRes=False)

# Remove tmp folders
VirtualLab.Cleanup()

