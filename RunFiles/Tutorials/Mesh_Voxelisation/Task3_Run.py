#!/usr/bin/env python3
#===============================================================================
# Header
#===============================================================================

import os
import sys
sys.dont_write_bytecode=True
import requests
from Scripts.Common.VirtualLab import VLSetup
import VLconfig

#===============================================================================
# Definitions
#===============================================================================

Simulation='Dragon'
Project='Tutorials'
Parameters_Master='TrainingParameters_Dragon'
Parameters_Var=None


# path to IBSim mesh file
stl_fname = "{}/Dragon/Tutorials/Meshes/welsh-dragon-small.stl".format(VLconfig.OutputDir)
if not os.path.isfile(stl_fname):
    os.makedirs(os.path.dirname(stl_fname),exist_ok=True)
    # Download file from link
    r = requests.get('https://sourceforge.net/p/gvirtualxray/code/HEAD/tree/trunk/SimpleGVXR-examples/WelshDragon/welsh-dragon-small.stl?format=raw')
    # write to file
    with open(stl_fname,'wb') as f:
        f.write(r.content)

#===============================================================================
# Environment
#===============================================================================

VirtualLab=VLSetup(
           Simulation,
           Project
           )

VirtualLab.Settings(
           Mode='Interactive',
           Launcher='Process',
           NbJobs=1
           )

VirtualLab.Parameters(
           Parameters_Master,
           Parameters_Var,
           RunMesh=False,
           RunSim=False,
           RunDA=False
           )

#===============================================================================
# Methods
#===============================================================================

VirtualLab.Voxelise()
