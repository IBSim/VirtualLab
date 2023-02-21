#!/usr/bin/env python3
#===============================================================================
# Header
#===============================================================================

import sys
import os
import requests
sys.dont_write_bytecode=True
from Scripts.Common.VirtualLab import VLSetup
import VLconfig

Simulation='Tensile'
Project='Tutorials'
Parameters_Master='TrainingParameters_IBSim'
Parameters_Var=None


# path to IBSim mesh file
mesh_fname = "{}/Tensile/Tutorials/Meshes/Tensile_IBSim.med".format(VLconfig.OutputDir)
if not os.path.isfile(mesh_fname):
    # Download file from link
    r = requests.get('https://ibsim.co.uk/VirtualLab/downloads/Tensile_IBSim.med')
    # write to file
    with open(mesh_fname,'wb') as f:
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
           Parameters_Var
           )

#===============================================================================
# Methods
#===============================================================================

VirtualLab.Mesh()

VirtualLab.Sim(
           ShowRes=True
           )

VirtualLab.DA()
