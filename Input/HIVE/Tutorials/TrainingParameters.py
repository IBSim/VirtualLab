from types import SimpleNamespace as Namespace
Mesh = Namespace()
Sim = Namespace()

 # This flag indicates whether we want to use ERMES to represent the thermal load generated by the coil or a uniform heat flux
EMLoad = 'Uniform' # {'Uniform','ERMES'}

##########################
######## Meshing #########
##########################

Mesh.Name = 'AMAZE_Sample'
Mesh.File = 'AMAZE' # This file must be in Scripts/$SIMULATION/PreProc
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
# Mesh parameters
Mesh.Length1D = 0.005
Mesh.Length2D = 0.005
Mesh.Length3D = 0.005
Mesh.PipeSegmentN = 20 # Number of segments for pipe circumference
Mesh.SubTile = 0.002 # Mesh fineness on tile

##########################
####### Simulation #######
##########################
Sim.Name = 'Sim_Uniform'

#############
## PreAster #
#############
Sim.PreAsterFile = "PreHIVE"
# HTC between coolant and pipe (need Coolant and Pipe properties)
Sim.CreateHTC = True
Sim.Pipe = {'Type':'smooth tube', 'Diameter':0.01, 'Length':0.05}
Sim.Coolant = {'Temperature':20, 'Pressure':2, 'Velocity':10}
# Pre-processing to create EMLoads from ERMES output
if EMLoad == 'ERMES':
    Sim.RunERMES = True
    Sim.CoilType = 'Test'
    Sim.CoilDisplacement = [0,0,0.0015]
    Sim.Rotation = 0

    Sim.Current = 1000
    Sim.Frequency = 1e4
    Sim.NbProc = 1

    Sim.Threshold = 1
    Sim.NbClusters = 100

#############
### Aster ###
#############
Sim.AsterFile = 'AMAZE_SS' # This file must be in Scripts/$SIMULATION/Aster
Sim.Mesh = 'AMAZE_Sample' # The mesh used for the simulation
Sim.Model = '3D'
Sim.Solver = 'MUMPS'
# Loading
Sim.EMLoad = EMLoad
if EMLoad == 'Uniform':
    Sim.Flux = 1e6

### Materials
Sim.Materials = {'Block':'Copper_NL', 'Pipe':'Copper_NL', 'Tile':'Tungsten_NL'}

#############
# PostAster #
#############
