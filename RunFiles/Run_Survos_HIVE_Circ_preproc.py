#!/usr/bin/env python3
#script to run full CT scan and reconstruction with HIVE mesh
import sys
sys.dont_write_bytecode=True
from Scripts.Common.VirtualLab import VLSetup

#===============================================================================
# Setup

Simulation='GVXR'
Project='Circular'
Parameters_Master='TrainingParameters_Survos-HIVE_preproc'
# Parameters_Var='TrainingParameters_Survos-HIVE_var'
Parameters_Var=None
#===============================================================================
# Environment

VirtualLab=VLSetup(
           Simulation,
           Project)

VirtualLab.Settings(
         #   Mode='Interactive',
           Mode='Headless',
           Launcher='Process',
           NbJobs=1)

VirtualLab.Parameters(
           Parameters_Master,
           Parameters_Var,
           RunMesh=True,
           RunSim=False,
           RunDA=False,
           RunVoxelise=True,
           RunCT_Scan=True,
           RunCT_Recon=True,
           RunImPreProc=True)
# Hive anlysis
VirtualLab.Mesh(
           ShowMesh=False,
           MeshCheck=None)

VirtualLab.DA()
#Voxelsisation
VirtualLab.Voxelise()
#GVXR
VirtualLab.CT_Scan()
#CIL
VirtualLab.CT_Recon()
# Reg/Normalisation
VirtualLab.ImPreProc(
  Normalise=True,
  Register=True)