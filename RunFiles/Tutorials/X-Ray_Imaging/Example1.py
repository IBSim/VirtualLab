#!/usr/bin/env python3

#===============================================================================
# Header

import os
import requests
import sys
sys.dont_write_bytecode=True
from Scripts.Common.VirtualLab import VLSetup


Simulation='GVXR'
Project='Tutorials'
Parameters_Master='TrainingParameters_GVXR-Draig'
Parameters_Var=None

#===============================================================================
# Environment
#===============================================================================

VirtualLab=VLSetup(
        Simulation,
        Project
        )

mesh_file = "{}/Meshes/welsh-dragon-small.stl".format(VirtualLab.PARAMETERS_DIR)
if not os.path.isfile(mesh_file):
    # Download file from link
    r = requests.get('https://sourceforge.net/p/gvirtualxray/code/HEAD/tree/trunk/SimpleGVXR-examples/WelshDragon/welsh-dragon-small.stl')
    # write to file
    os.makedirs(os.path.dirname(mesh_file),exist_ok=True)
    with open(mesh_file,'wb') as f:
        f.write(r.content)

VirtualLab.Settings(
        Mode='Interactive',
        Launcher='Process',
        NbJobs=1
        )

VirtualLab.Parameters(
        Parameters_Master,
        Parameters_Var,
        RunCT_Scan=True,
        RunCT_Recon=False
        )

VirtualLab.CT_Scan()