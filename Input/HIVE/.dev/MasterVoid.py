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
Mesh.File = 'AMAZE_Void' # This file must be in Scripts/$SIMULATION/PreProc
# Geometrical Dimensions for Fundamental Hive Sample
Mesh.BlockWidth = 0.045 #x
Mesh.BlockLength = 0.045 #y
Mesh.BlockHeight = 0.035 #z
Mesh.PipeDiam = 0.0127 #Pipe inner diameter
Mesh.PipeThick = 0.00415 #Pipe wall thickness
Mesh.PipeLength = 0.20
Mesh.TileCentre = [0,0]
Mesh.TileWidth = Mesh.BlockWidth
Mesh.TileLength = 0.045 #y
Mesh.TileHeight = 0.010 #z
Mesh.PipeCentre = [0, Mesh.TileHeight/2]

# =============================================================================
# Geometrical Dimensions for Single Void in Elliptic Cylinder Geometry
# Void centre/location
Mesh.VoidCentre = [[1.0/12.0, 1.0/6.0], [3.0/12.0, 1/6.0]] # in terms of tile corner [1.0/6.0 -0.00125*2.0, 1.0/6.0 -0.00125*2.0]
Mesh.Void = [[0.00125, 0.0005, 0.005, 0.0], [0.00125, 0.0005, 0.005, 0.0]] #[0.002, 0.002, 0.002, 0.0 ], [0.002, 0.003, 0.002, 0.0 ], [0.001, 0.002, 0.001, 60.0 ]

# Mesh.Void = [[VoidRad_a, VoidRad_b, VoidHeight, VoidRotation], [...], ...]
# VoidRad_a: length of void in horizontal axis X
# VoidRad_b: width of void in vertical axis Y
# VoidHeight: height of void < TileHeigth
# VoidRotation: amount of rotation in degree 
# =============================================================================
# Mesh parameters
Mesh.Length1D = 0.0035
Mesh.Length2D = 0.0035
Mesh.Length3D = 0.0035

Mesh.SubTile = 0.002 # Mesh fineness on tile
Mesh.PipeSegmentN = 24 # Number of segments for pipe circumference
Mesh.VoidSegmentN = 16 # Number of segments for hole circumference (for sub-mesh) 16-24-28-32..



##########################
####### Simulation #######
##########################
Sim.Name = 'MultiVoid' #  no void flag: 'NoVoid', single void flag: 'SingleVoid', multi void flag: 'MultiVoid' 

#############
## PreAster #
#############
Sim.PreAsterFile = "devPreHIVE"
# HTC between coolant and pipe (need Coolant and Pipe properties)
Sim.CreateHTC = True
Sim.Pipe = {'Type':'smooth tube', 'Diameter':Mesh.PipeDiam, 'Length':Mesh.PipeLength}
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
Sim.Materials = {'Block':'Ti6Al4V_Annealed', 'Pipe':'Ti6Al4V_Annealed', 'Tile':'Ti6Al4V_Annealed'}

### IC - Need either an initial temperature or a results file to import
Sim.ImportRes = False
Sim.InitTemp = 25 #Celcius

### Time-stepping and temporal discretisation
Sim.Theta = 0.5
Sim.dt = [(0.01,10,2), (0.05,200,2), (0.1,100,2), (0.5,100,2), (1,50,2), (5,50,2) ] #timestep size and number of steps [(0.01,200,2)]  => 99% temp is reached at [(0.01,200,2), (0.2,1650,5)]

Sim.Convergence = {'Start':10,'Gap':5}


#############
# PostAster #
#############

Sim.PostAsterFile = 'HIVEPost'

## if there are any thermocouples, define their location in terms of solid, surface and position
Sim.ThermoCouple = [ ['Tile', 'Front', 1.0/6.0, 3.0/6.0],  ['Tile', 'Front', 3.0/6.0, 3.0/6.0],  ['Tile', 'Front', 5.0/6.0, 3.0/6.0], ['Tile', 'Back', 1.0/6.0, 3.0/6.0],  ['Tile', 'Back', 3.0/6.0, 3.0/6.0], ['Tile', 'Back', 5.0/6.0, 3.0/6.0], ['Tile', 'SideA', 1.0/6.0, 3.0/6.0],  ['Tile', 'SideA', 3.0/6.0, 3.0/6.0], ['Tile', 'SideA', 5.0/6.0, 3.0/6.0], ['Tile', 'SideB', 1.0/6.0, 3.0/6.0],  ['Tile', 'SideB', 3.0/6.0, 3.0/6.0],  ['Tile', 'SideB', 5.0/6.0, 3.0/6.0], ['Block', 'Front', 1.0/6.0, 1.0/6.0], ['Block', 'Front', 3.0/6.0, 1.0/6.0], ['Block', 'Front', 5.0/6.0, 1.0/6.0], ['Block', 'Front', 1.0/6.0, 3.0/6.0],  ['Block', 'Front', 3.0/6.0, 3.0/6.0],  ['Block', 'Front', 5.0/6.0, 3.0/6.0],  ['Block', 'Front', 1.0/6.0, 5.0/6.0], ['Block', 'Front', 3.0/6.0, 5.0/6.0], ['Block', 'Front', 5.0/6.0, 5.0/6.0], ['Block', 'Back', 1.0/6.0, 1.0/6.0], ['Block', 'Back', 3.0/6.0, 1.0/6.0], ['Block', 'Back', 5.0/6.0, 1.0/6.0], ['Block', 'Back', 1.0/6.0, 3.0/6.0],  ['Block', 'Back', 3.0/6.0, 3.0/6.0], ['Block', 'Back', 5.0/6.0, 3.0/6.0],  ['Block', 'Back', 1.0/6.0, 5.0/6.0], ['Block', 'Back', 3.0/6.0, 5.0/6.0], ['Block', 'Back', 5.0/6.0, 5.0/6.0], ['Block', 'SideA', 1.0/6.0, 1.0/6.0], ['Block', 'SideA', 3.0/6.0, 1.0/6.0], ['Block', 'SideA', 5.0/6.0, 1.0/6.0], ['Block', 'SideA', 1.0/6.0, 3.0/6.0], ['Block', 'SideA', 5.0/6.0, 3.0/6.0],  ['Block', 'SideA', 1.0/6.0, 5.0/6.0], ['Block', 'SideA', 5.0/6.0, 5.0/6.0], ['Block', 'SideB', 1.0/6.0, 1.0/6.0], ['Block', 'SideB', 3.0/6.0, 1.0/6.0], ['Block', 'SideB', 5.0/6.0, 1.0/6.0], ['Block', 'SideB', 1.0/6.0, 3.0/6.0], ['Block', 'SideB', 5.0/6.0, 3.0/6.0],  ['Block', 'SideB', 1.0/6.0, 5.0/6.0], ['Block', 'SideB', 5.0/6.0, 5.0/6.0], ['Block', 'Bottom', 1.0/6.0, 1.0/6.0], ['Block', 'Bottom', 3.0/6.0, 1.0/6.0], ['Block', 'Bottom', 5.0/6.0, 1.0/6.0], ['Block', 'Bottom', 1.0/6.0, 3.0/6.0], ['Block', 'Bottom', 3.0/6.0, 3.0/6.0], ['Block', 'Bottom', 5.0/6.0, 3.0/6.0], ['Block', 'Bottom', 1.0/6.0, 5.0/6.0], ['Block', 'Bottom', 3.0/6.0, 5.0/6.0], ['Block', 'Bottom', 5.0/6.0, 5.0/6.0]]
# Solid flags: 'Tile', 'Block'
# Surface flags: (for Tile) 'Front', 'Back','SideA', 'SideB', 'Top'; (for Block) 'Front', 'Back', 'SideA', 'SideB', 'Bottom' 

# in normalised local coordinates: topsurface respectively btw 0.0 and 1.0
# (0.0, 0.0): left-bottom corner, (1.0, 1.0): right-top corner
Sim.Rvalues = [0.0025]# Radii of the search; multi thermocouple:; [0.001, 0.0025, 0.005]

Sim.CaptureTime = 0.1 #"all" # or 2.0 for all increments
Sim.TemperatureOut = True # if you want to write average temperature data on thermocouples in a file ('ThermocoupleTemp.txt')
Sim.TemperaturePlot = False # if you want to plot average temperature data on thermocouples in a single plot
ML.Name = 'Test'
ML.File = 'NetPU'
