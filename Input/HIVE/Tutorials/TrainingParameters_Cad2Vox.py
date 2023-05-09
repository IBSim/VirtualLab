from types import SimpleNamespace as Namespace
Mesh = Namespace()
Sim = Namespace()

##########################
######## Meshing #########
##########################

Mesh.Name = 'AMAZE'
Mesh.File = 'Monoblock' # This file must be in Scripts/$SIMULATION/PreProc
# Geometrical Dimensions
Mesh.BlockWidth = 0.03 #x
Mesh.BlockLength = 0.05 #y
Mesh.BlockHeight = 0.02 #z
Mesh.PipeCentre = (0,0) #x,z, relative to centre of block
Mesh.PipeDiam = 0.01 ###Inner Diameter
Mesh.PipeThick = 0.001
Mesh.PipeLength = Mesh.BlockLength
Mesh.TileCentre = (0,0)
Mesh.TileWidth = Mesh.BlockWidth
Mesh.TileLength = 0.03 #y
Mesh.TileHeight = 0.005 #z

Mesh.Fillet = 0.0005

# Mesh parameters
Mesh.Length1D = 0.005
Mesh.Length2D = 0.005
Mesh.Length3D = 0.005

Mesh.PipeSegmentN = 20 # Number of segments for pipe circumference
Mesh.SubTile = [0.002, 0.002, 0.002]
Mesh.Deflection = 0.01

##########################
####### Simulation #######
##########################
Sim.Name = 'Examples/Test_Coil'
Sim.Mesh = 'AMAZE' # The mesh used for the simulation
Sim.Materials = {'Block':'Copper_NL', 'Pipe':'Copper_NL', 'Tile':'Tungsten_NL'}

#############
## Coolant ##
#############
# HTC between coolant and pipe (need Coolant and Pipe properties)
Sim.Pipe = {'Type':'smooth tube', 'Diameter':0.01, 'Length':0.05}
Sim.Coolant = {'Temperature':30, 'Pressure':1, 'Velocity':10}
# Pre-processing to create EMLoads from ERMES output

#############
### ERMES ###
#############
Sim.CoilType = 'Test'
Sim.CoilDisplacement = [0,0,0.0015]
Sim.Frequency = 1e4

#############
### Aster ###
#############
Sim.AsterFile = 'Monoblock_Steady' # This file must be in Scripts/$SIMULATION/Aster
Sim.Model = '3D'
Sim.Solver = 'MUMPS'

Sim.Current = 1000
Sim.NbClusters = 100

#########################
###### Voxelisation #####
##########################
Vox = Namespace()
# name for the cad2vox run.
# Note: this doubles up as the output file name
Vox.Name = 'AMAZE'
# name of the mesh(es) you wish to voxelise
Vox.mesh = 'AMAZE'
# Number of voxels in each dimension
Vox.gridsize = [250,250,250]

#### Optional Arguments #############
# Skip the check for GPU and fallback to CPU
Vox.cpu = True
# Force the use of Tetrahedron data instead of the default Triangles
Vox.use_tetra = True