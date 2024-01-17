#!/usr/bin/env python3
import sys
sys.dont_write_bytecode=True
from Scripts.Common.VirtualLab import VLSetup

# =============================================================================
# Create VirtualLab instance & define settings

Simulation='Irradiation'
Project='Tutorials'

VirtualLab=VLSetup(
           Simulation,
           Project)

VirtualLab.Settings(
           Mode='T',
           )

# =============================================================================
# Perform neutronics simulation and obtain neutron heat and damage energy loads.
Parameters_Master='neutron_heat'
Parameters_Var=None

VirtualLab.Parameters(
           Parameters_Master,
           Parameters_Var,
           RunParamak=True,
           RunOpenmc=True,
           Runparaview=True,
           RunMesh=True,
           RunSim=True
           )
           
VirtualLab.Paramak()

VirtualLab.Openmc()

VirtualLab.paraview()

VirtualLab.Mesh(
           ShowMesh=False,
           MeshCheck=None)

VirtualLab.Sim(RunPreAster=True,
           RunCoolant=True,
           RunERMES=False,
           RunAster=True,
           RunPostAster=False,
           ShowRes=False)     

# =============================================================================
# Convert damage energy loads from neutronics simulation and calculate dpa.

Parameters_Master='damage'
Parameters_Var=None

VirtualLab.Parameters(
           Parameters_Master,
           Parameters_Var,
           RunSim=True,
           RunDPA=True,
           )
           
VirtualLab.Sim(RunPreAster=False,
           RunCoolant=False,
           RunERMES=False,
           RunAster=True,
           RunPostAster=False,
           ShowRes=False)     
           
           
VirtualLab.DPA()

# =============================================================================           
# Convert dpa into finite element mesh.

Parameters_Master='dpa_post'
Parameters_Var=None

VirtualLab.Parameters(
           Parameters_Master,
           Parameters_Var,
           RunSim=True,
           )
           
VirtualLab.Sim(
           RunPreAster=False,
           RunCoolant=False,
           RunAster=True,
           RunERMES=False,
           RunPostAster=False,
           ShowRes=False)
           

# =============================================================================           
# Thermal analysis

Parameters_Master='thermal'
Parameters_Var=None

VirtualLab.Parameters(
           Parameters_Master,
           Parameters_Var,
           RunSim=True,
           )
           

VirtualLab.Sim( RunPreAster=False,
           RunCoolant=True,
           RunERMES=False,
           RunAster=True,
           RunPostAster=False,
           ShowRes=False)

# =============================================================================           
# Mechanical analysis

Parameters_Master='mechanical'
Parameters_Var=None

VirtualLab.Parameters(
           Parameters_Master,
           Parameters_Var,
           RunSim=True,
           )
           
VirtualLab.Sim( RunPreAster=False,
           RunCoolant=True,
           RunERMES=False,
           RunAster=True,
           RunPostAster=False,
           ShowRes=False)

# =============================================================================           
# Lifecycle analysis

Parameters_Master='lifecycle'
Parameters_Var=None

VirtualLab.Parameters(
           Parameters_Master,
           Parameters_Var,
           RunDA=True)
           
VirtualLab.DA()

# =============================================================================           
# Lifecycle analysis

Parameters_Master='lifecycle1'
Parameters_Var=None

VirtualLab.Parameters(
           Parameters_Master,
           Parameters_Var,
           Runlifecycle=True,
           RunDA=True)

VirtualLab.lifecycle()           

VirtualLab.DA()

       
