from types import SimpleNamespace as Namespace

##########################
######## Meshing #########
##########################
Mesh = Namespace()
Mesh.Name = 'HiveCoil'
Mesh.File = 'AMAZE' # This file must be in Scripts/$SIMULATION/PreProc
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
Mesh.CircDisc = 20 # Number of segments for pipe circumference
Mesh.Sub2_1D = 0.001 # Mesh fineness on tile 

Mesh.CoilType = 'Test'
Mesh.CoilDisp = [0, 0, 0.002]

##########################
####### Simulation #######
##########################
Sim = Namespace()
Sim.Name = 'Single'

#############
## Pre-Sim ##
#############
## ERMES solver is used to creat BC for induction heating
Sim.RunEM = 'N'

Sim.Frequency = 1e4
Sim.Current = 10 #Current in the coil
Sim.NProc = 2

## HTC between cooland and pipe
Sim.CreateHTC = 'N' 
Sim.PipeGeom = 'smooth tube'
Sim.FluidT = 20 #Celcius
Sim.FluidP = 1 #MPa
Sim.FluidV = 10 #m/s

#############
### Aster ###
#############
Sim.CommFile = 'AMAZE_Adapt' # This file must be in Scripts/$SIMULATION/Aster
Sim.Mesh = 'HiveCoil' # The mesh used for the simulation

Sim.Model = '3D'
Sim.Solver = 'MUMPS'

### Materials
Sim.Materials = {'Block':'Copper_NL', 'Pipe':'Copper_NL', 'Tile':'Tungsten_NL'}

### IC - Need either an initial temperature or a results file to import
Sim.ImportRes = 'n'
Sim.InitTemp = 20 #Celcius

### Time-stepping and temporal discretisation
Sim.Theta = 0.5
Sim.dt = [(0.001,10)] #timestep size and number of steps
Sim.ResStore = 3 #How often should results be stored
Sim.CheckFirst = 3
Sim.CheckEvery = 3 #If this is smaller than ResStore then ResStore will be used

#############
## Post-Sim #
#############

