from types import SimpleNamespace as Namespace

##########################
######## Meshing #########
##########################
Mesh = Namespace()
Mesh.Name = 'Notch1'
Mesh.File = 'DogBone'
# Geometric parameters (all units in metres)
Mesh.Thickness = 0.003
Mesh.HandleWidth = 0.024
Mesh.HandleLength = 0.024
Mesh.GaugeWidth = 0.012
Mesh.GaugeLength = 0.04
Mesh.TransRad = 0.012
Mesh.HoleCentre = (0.0,0.0)
Mesh.Rad_a = 0.001
Mesh.Rad_b = 0.0005
# Parameters to generate mesh
Mesh.Length1D = 0.001
Mesh.Length2D = 0.001
Mesh.Length3D = 0.001
Mesh.HoleDisc = 30 # Number of segments for hole circumference (for sub-mesh)

##########################
####### Simulation #######
##########################
Sim = Namespace()
Sim.Name = 'Single'
#############
## Pre-Sim ##
#############

#############
### Aster ###
#############
'''
A tensile test can be conducted either by constant force or constant displacement.
The key(s) in the 'Load' dictionary will dictate the type of loading used.
Force is measured in N, Displacement in M.
'''
Sim.AsterFile = 'Tensile' # The CodeAster command file can be found in Scripts/$SIMULATION/Aster
Sim.Mesh = 'Notch1' # The mesh used in the simulation
Sim.Load = {'Force':1000000, 'Displacement':0.01}
Sim.ResName = ['ConstForce', 'ConstDisp'] # Name of results file(s) which will be output by CodeAster
# Material type(s) for analysis, the properties of which can be found in the 'Materials' directory
Sim.Materials = 'Copper'

#############
## Post-Sim #
#############
