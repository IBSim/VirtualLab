from types import SimpleNamespace as Namespace
from Scripts.Common.VLFunctions import ParametersVar

##########################
######## Meshing #########
##########################
Mesh = Namespace()
# Name under which the mesh will be saved in Meshes directory.
Mesh.Name = ParametersVar(['Notch2','Notch3'])
# Salome python file used to create mesh.
Mesh.File = 'DogBone'

# Geometric Parameters
Mesh.Thickness = 0.003
Mesh.HandleWidth = 0.024
Mesh.HandleLength = 0.024
Mesh.GaugeWidth = 0.012
Mesh.GaugeLength = 0.04
Mesh.TransRad = 0.012
# Optional parameters to add hole to sample
Mesh.HoleCentre = (0.0,0.0)
Mesh.Rad_a = ParametersVar([0.001,0.002])
Mesh.Rad_b = ParametersVar([0.001,0.0005])

# Meshing Parameters
# Discretisation along edges (1D)
Mesh.Length1D = 0.001
# Discretisation on faces (2D)
Mesh.Length2D = 0.001
# Discretisation on volumes (3D)
Mesh.Length3D = 0.001
# Number of segments for hole circumference
Mesh.HoleSegmentN = 30

##########################
####### Simulation #######
##########################
Sim = Namespace()

# Name under which the simulation results will be saved.
Sim.Name = ParametersVar(['ParametricSim1', 'ParametricSim2'])
# The CodeAster command file can be found in Scripts/$SIMULATION.
Sim.AsterFile = 'Tensile'

# The mesh used in the simulation.
Sim.Mesh = ParametersVar(['Notch2', 'Notch3'])
# Force applied in force controlled analysis.
Sim.Force = 1000000
# Enforced displacement in displacement controlled analysis.
Sim.Displacement = 0.01
# Material specimen is made of. Properties can be found in the 'Materials' directory.
Sim.Materials = 'Copper'
