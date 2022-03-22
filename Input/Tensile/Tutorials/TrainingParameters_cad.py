from types import SimpleNamespace as Namespace

##########################
######## Meshing #########
##########################
Mesh = Namespace()

# Name under which the mesh will be saved in Meshes directory.
Mesh.Name = 'Notch1'
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
Mesh.Rad_a = 0.0005
Mesh.Rad_b = 0.001

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
Sim.Name = 'Single'
# The CodeAster command file can be found in Scripts/$SIMULATION.
Sim.AsterFile = 'Tensile'

# The mesh used in the simulation.
Sim.Mesh = 'Notch1'
# Force applied in force controlled analysis.
Sim.Force = 1000000
# Enforced displacement in displacement controlled analysis.
Sim.Displacement = 0.01
# Material specimen is made of. Properties can be found in the 'Materials' directory.
Sim.Materials = 'Copper'

#########################
###### Voxelisation #####
##########################
Vox = Namespace()
# names of the meshes you wish to voxelise
Vox.Name = 'Notch1'
Vox.gridsize = 250
Vox.cpu = True
Vox.use_tetra = True