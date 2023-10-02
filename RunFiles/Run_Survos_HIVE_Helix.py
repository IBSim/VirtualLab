#!/usr/bin/env python3
#script to run full CT scan and reconstruction with HIVE mesh
import sys
sys.dont_write_bytecode=True
from Scripts.Common.VirtualLab import VLSetup

#===============================================================================
# Setup

Simulation='GVXR'
Project='Helix'
Parameters_Master='TrainingParameters_Survos-HIVE'
Parameters_Var='TrainingParameters_Survos-HIVE_var1'
#===============================================================================
# Environment

VirtualLab=VLSetup(
           Simulation,
           Project)

VirtualLab.Settings(
           Mode='Headless',
           Launcher='Process',
           NbJobs=10)

VirtualLab.Parameters(
           Parameters_Master,
           Parameters_Var,
           RunMesh=False,
           RunSim=False,
           RunDA=False,
           RunVoxelise=False,
           RunCT_Scan=False,
           RunCT_Recon=False,
           RunImPreProc=True)
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
# image preproc
VirtualLab.ImPreProc(
    Normalise=True,
    Register=False,
    Helix=True,
)