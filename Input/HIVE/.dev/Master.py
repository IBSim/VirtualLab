from types import SimpleNamespace as Namespace
Mesh = Namespace()
Sim = Namespace()
ML = Namespace()

EMLoad = 'ERMES'

##########################
######## Meshing #########
##########################

Mesh.Name = 'Mesh_void'
Mesh.File = 'AMAZE_SingleVoid' # This file must be in Scripts/$SIMULATION/PreProc
# Geometrical Dimensions
Mesh.BlockWidth = 0.03 #x
Mesh.BlockLength = 0.05 #y
Mesh.BlockHeight = 0.02 #z
Mesh.PipeCentre = [0,0] #x,z, relative to centre of block
Mesh.PipeDiam = 0.01 ###Inner Diameter
Mesh.PipeThick = 0.001
Mesh.PipeLength = Mesh.BlockLength
Mesh.TileCentre = [0,0]
Mesh.TileWidth = Mesh.BlockWidth
Mesh.TileLength = 0.03 #y
Mesh.TileHeight = 0.005 #z
# Mesh parameters
Mesh.Length1D = 0.005
Mesh.Length2D = 0.005
Mesh.Length3D = 0.005

Mesh.SubTile = 0.002 # Mesh fineness on tile

Mesh.VoidCentre = (Mesh.BlockWidth*0.5 ,Mesh.BlockLength*0.5) # Void centre relative to centre of block - (0, 0) is at the centre
Mesh.VoidDiam = 0.001 # Diameter of void
Mesh.VoidHeight = 0.001 # Height of Void. Positive/negative number gives a void in the top/bottom
###############################################################################
# Mesh parameters
Mesh.PipeSegmentN = 20 # Number of segments for pipe circumference
Mesh.VoidSegmentN = 32 # Number of segments for hole circumference (for sub-mesh) (ES) 16-24-28-32..


##########################
####### Simulation #######
##########################
Sim.Name = 'Test_Meshvoid'

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


    Sim.Threshold = 0.9
    Sim.ThresholdPreview = False


#############
### Aster ###
#############
Sim.AsterFile = 'AMAZE' # This file must be in Scripts/$SIMULATION/Aster
Sim.Mesh = Mesh.Name # The mesh used for the simulation
Sim.Model = '3D'
Sim.Solver = 'MUMPS'
# Loading
Sim.EMLoad = EMLoad
if EMLoad == 'Uniform':
    Sim.Flux = 1e7

### Materials
Sim.Materials = {'Block':'Copper_NL', 'Pipe':'Copper_NL', 'Tile':'Tungsten_NL'}

### IC - Need either an initial temperature or a results file to import
Sim.ImportRes = False
Sim.InitTemp = 20 #Celcius

### Time-stepping and temporal discretisation
Sim.Theta = 0.5
Sim.dt = [(0.0001,20,1)] #timestep size and number of steps

Sim.Convergence = {'Start':10,'Gap':5}


#############
# PostAster #
#############

##########################
#### Machine Learning ####
##########################

ML.Name = 'Test'
ML.File = 'NetPU'

ML.NewData = True
ML.Device = 'cpu'
ML.DataPrcnt = 1
ML.SplitRatio = 0.7
ML.ShowMetric = 1
