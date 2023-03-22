#!/usr/bin/env python3
#script to run full CT scan and reconstruction with HIVE mesh
import sys
sys.dont_write_bytecode=True
from Scripts.Common.VirtualLab import VLSetup

#===============================================================================
# Setup

Simulation='GVXR'
Project='Survos'
Parameters_Master='TrainingParameters_Survos-HIVE'
Parameters_Var='TrainingParameters_Survos-HIVE_var'
#===============================================================================
# Environment

VirtualLab=VLSetup(
           Simulation,
           Project)

VirtualLab.Settings(
           Mode='Interactive',
           Launcher='Sequential',
           NbJobs=1)

VirtualLab.Parameters(
           Parameters_Master,
           Parameters_Var,
           RunMesh=False,
           RunSim=False,
           RunDA=False,
           RunVoxelise=False,
           RunCT_Scan=True,
           RunCT_Recon=False)
# Hive anlysis
VirtualLab.Mesh(
           ShowMesh=False,
           MeshCheck=None)

VirtualLab.Sim(
           RunPreAster=True,
           RunAster=True,
           RunPostAster=True,
           ShowRes=True)

VirtualLab.DA()
#Voxelsisation
VirtualLab.Voxelise()
#GVXR
VirtualLab.CT_Scan()
#CIL
VirtualLab.CT_Recon()
