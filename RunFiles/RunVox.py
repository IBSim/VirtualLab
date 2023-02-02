#!/usr/bin/env python3
import sys
sys.dont_write_bytecode=True
from Scripts.Common.VirtualLab import VLSetup

#===============================================================================
# Setup
Simulation='Dragon'
Project='Tutorials'
Parameters_Master='TrainingParameters_Dragon'
Parameters_Var=None

VirtualLab=VLSetup(
                Simulation,
                Project)

VirtualLab.Settings(
            Mode='Interactive',
            Launcher='Process',
            NbJobs=1)

VirtualLab.Parameters(
            Parameters_Master,
            Parameters_Var,
            RunMesh=False,
            RunSim=False,
            RunDA=False)

VirtualLab.Voxelise()