#!/usr/bin/env python3

import sys
from os.path import dirname, abspath
sys.dont_write_bytecode=True
sys.path.append(dirname(dirname(abspath(__file__))))
from Scripts.Common.VirtualLab import VLSetup

#try: from Scripts.Common import VLSetup
#except ModuleNotFoundError: 
#	sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#	from Scripts.Common import VLSetup		

Simulation = 'Tensile'
Project = 'Example'
StudyName = 'Training'
Parameters_Master='TrainingParameters'
Parameters_Var='Parametric_1'

# kwarg 'mode' has 3 options - interactive, continuous or headless (default)
VirtualLab = VLSetup(Simulation, Project, StudyName, Parameters_Master,Parameters_Var, mode = "Interactive")

# Create directories and Parameter files for simulation
VirtualLab.Create()

# Creates meshes
VirtualLab.Mesh()

# Run Pre-Sim calculations, CodeAster and Post-Sim calculations/imaging
VirtualLab.Sim()

# Remove tmp folders
VirtualLab.Cleanup()

