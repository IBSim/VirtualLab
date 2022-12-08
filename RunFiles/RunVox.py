#!/usr/bin/env python3
import sys
sys.dont_write_bytecode=True
from Scripts.Common.VirtualLab import VLSetup

Simulation='Tensile'
Project='Tutorials'
Parameters_Master='TrainingParameters_Cad2Vox'
Parameters_Var=None

VirtualLab = VLSetup(Simulation,Project)

VirtualLab.Settings(Mode='Interactive',Launcher='Process',NbJobs=1)

VirtualLab.Parameters(
            Parameters_Master,
            Parameters_Var,
            RunMesh=True,
            RunSim=True,
            RunDA=True,
            RunVoxelise=True)

VirtualLab.Mesh(
            ShowMesh=False,
            MeshCheck=None)

VirtualLab.Sim(
            RunPreAster=True,
            RunAster=True,
            RunPostAster=True,
            ShowRes=True)

VirtualLab.DA()
VirtualLab.Voxelise()
