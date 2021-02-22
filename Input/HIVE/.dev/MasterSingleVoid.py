from types import SimpleNamespace as Namespace
Mesh = Namespace()
Sim = Namespace()
ML = Namespace()

Mesh.Name = 'SingleVoid'  # if no void, 'NoVoid', 
                          # if single void, 'SingleVoid',
                          # if Multiple void, 'MultipleVoid'
EMLoad = 'ERMES'

##########################
######## Meshing #########
##########################

Mesh.Name = 'Mesh_SingleVoid'
Mesh.File = 'AMAZE_SingleVoid' # This file must be in Scripts/$SIMULATION/PreProc
# Geometrical Dimensions for Fundamental Hive Sample
Mesh.BlockWidth = 0.03 #x
Mesh.BlockLength = 0.05 #y
Mesh.BlockHeight = 0.02 #z
Mesh.PipeCentre = [0,0] #x,z, relative to centre of block
Mesh.PipeDiam = 0.01 #Pipe inner diameter
Mesh.PipeThick = 0.001 #Pipe wall thickness
Mesh.PipeLength = Mesh.BlockLength
Mesh.TileCentre = [0,0]
Mesh.TileWidth = Mesh.BlockWidth
Mesh.TileLength = 0.03 #y
Mesh.TileHeight = 0.005 #z

###############################################################################
# Geometrical Dimensions for Single Void in Elliptical Cylinder Geometry
# Void centre/location is a geometric parameter!!
Mesh.VoidCentre = (Mesh.BlockWidth*0.5 ,Mesh.BlockLength*0.5) # Void centre relative to centre of block - (0, 0) is at the centre
Mesh.VoidRad_a = 0.003 # length of void in horizontal axis X
Mesh.VoidRad_b = 0.002 # width of void in vertical axis Y
Mesh.VoidHeight = 0.001 # height of void < TileHeigth
Mesh.VoidRotation = 45 # amount of rotation in degree 
###############################################################################
# Mesh parameters
Mesh.Length1D = 0.005
Mesh.Length2D = 0.005
Mesh.Length3D = 0.005

Mesh.SubTile = 0.002 # Mesh fineness on tile
Mesh.PipeSegmentN = 32 # Number of segments for pipe circumference
Mesh.VoidSegmentN = 8 # Number of segments for hole circumference (for sub-mesh) 16-24-28-32..


##########################
####### Simulation #######
##########################
Sim.Name = 'SingleVoid' # if no void, then use the flag'Single' 

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


ML.Name = 'Test'
ML.File = 'NetPU'
