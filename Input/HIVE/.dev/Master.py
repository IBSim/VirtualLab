from types import SimpleNamespace as Namespace
Mesh = Namespace()
Sim = Namespace()
ML = Namespace()

EMLoad = 'Uniform'

##########################
######## Meshing #########
##########################

Mesh.Name = 'AMAZEMesh'
Mesh.File = 'AMAZE' # This file must be in Scripts/$SIMULATION/PreProc
# Geometrical Dimensions
Mesh.BlockWidth = 0.03 #x
Mesh.BlockLength = 0.05 #y
Mesh.BlockHeight = 0.02 #z
Mesh.PipeCentre = [0,0] #x,z, relative to centre of block
Mesh.PipeDiam = 0.01 ###Inner Diameter
Mesh.PipeThick = 0.001
Mesh.PipeLength = Mesh.BlockLength*2
Mesh.TileCentre = [0,0]
Mesh.TileWidth = Mesh.BlockWidth
Mesh.TileLength = 0.03 #y
Mesh.TileHeight = 0.005 #z

Mesh.VoidCentre = [[0.25, 0.25],[0.75, 0.75]] # in terms of tile corner
Mesh.Void = [[0.001, 0.002, 0.004, 0.0],[0.002, 0.001, 0.004, 0.0]]

# Mesh parameters
Mesh.Length1D = 0.005
Mesh.Length2D = 0.005
Mesh.Length3D = 0.005
Mesh.SubTile = 0.002 # Mesh fineness on tile
Mesh.PipeSegmentN = 20 # Number of segments for pipe circumference
Mesh.VoidSegmentN = 32 # Number of segments for hole circumference (for sub-mesh) (ES) 16-24-28-32..


##########################
####### Simulation #######
##########################
Sim.Name = 'Test_1'

#############
## PreAster #
#############
Sim.PreAsterFile = "devPreHIVE"
# HTC between coolant and pipe (need Coolant and Pipe properties)
Sim.CreateHTC = True
Sim.Pipe = {'Type':'smooth tube', 'Diameter':0.01, 'Length':0.05}
Sim.Coolant = {'Temperature':20, 'Pressure':2, 'Velocity':10}
# Pre-processing to create EMLoads using ERMES
if EMLoad == 'ERMES':
    Sim.RunERMES = 1 #Boolean flag to run ERMES or not
    # ERMES setup - coil type & location
    Sim.CoilType = 'HIVE'
    Sim.CoilDisplacement = [0,0,0.0015]
    Sim.Rotation = 0
    Sim.VacuumRadius = 0.2
    # ERMES meshing parameters
    # Coil
    Sim.Coil1D = 0.001
    Sim.Coil2D = 0.001
    Sim.Coil3D = 0.001
    # Vacuum
    Sim.VacuumSegment = 25
    # Sim.Vacuum1D = 1
    # Sim.Vacuum2D = 1
    # Sim.Vacuum3D = 1

    Sim.NbProc = 1
    Sim.Current = 1000
    Sim.Frequency = 1e4


    Sim.Threshold = 1
    Sim.ThresholdPreview = False
    Sim.NbClusters = 100


#############
### Aster ###
#############
Sim.AsterFile = 'AMAZE_SS' # This file must be in Scripts/$SIMULATION/Aster
Sim.Mesh = Mesh.Name # The mesh used for the simulation
Sim.Model = '3D'
Sim.Solver = 'MUMPS'
# Loading
Sim.EMLoad = EMLoad
if EMLoad == 'Uniform':
    Sim.Flux = 1e7

Sim.BlockPipeTCC=None
Sim.TileBlockTCC=None

Sim.BlockEmissivity=0
Sim.TileEmissivity=0

### Materials
Sim.Materials = {'Block':'Copper_NL', 'Pipe':'Copper_NL', 'Tile':'Tungsten_NL'}

### IC - Need either an initial temperature or a results file to import
Sim.ImportRes = False
Sim.InitTemp = 20 #Celcius

### Time-stepping and temporal discretisation
Sim.Theta = 0.5
Sim.dt = [(0.002,100,1)] #timestep size and number of steps

Sim.Convergence = {'Start':10,'Gap':5}
