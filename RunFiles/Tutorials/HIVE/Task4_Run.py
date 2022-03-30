#!/usr/bin/env python3
#===============================================================================
# Header
import sys
sys.dont_write_bytecode=True
from Scripts.Common.VirtualLab import VLSetup

#===============================================================================
# Setup

Simulation='HIVE'
Project='Tutorials'
Parameters_Master='TrainingParameters_Task4'
Parameters_Var=None


#===============================================================================
# Environment

VirtualLab=VLSetup(
           Simulation,
           Project)

# Copy files created in task 4
import shutil
import os
NewResDir = "{}/Examples/ERMES_2".format(VirtualLab.PROJECT_DIR)
if os.path.isdir(NewResDir): shutil.rmtree(NewResDir)
shutil.copytree("{}/Examples/ERMES".format(VirtualLab.PROJECT_DIR), NewResDir)

VirtualLab.Settings(
           Mode='Interactive',
           Launcher='Process',
           NbJobs=1)

VirtualLab.Parameters(
           Parameters_Master,
           Parameters_Var,
           RunMesh=False)

VirtualLab.Mesh()

VirtualLab.Sim(ShowRes=True)

VirtualLab.DA()
