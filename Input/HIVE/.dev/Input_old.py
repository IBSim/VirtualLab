
##########################
##### Pre-processing #####
##########################
'''The mesh which will be used for the study should be input at 'MeshName', however if 'CreateMesh' 
is set to Yes then this name will be assigned to the mesh created using the dimensions defined below'''

CreateMesh = 'N'
MeshName = 'HiveCoil'

## Parameters used to create mesh if the flag is set to 'Y'
MeshFile = 'AMAZE'
# Geometrical Dimensions
BlockWidth = 0.03 #x
BlockLength = 0.05 #y
BlockHeight = 0.02 #z
PipeCentre = [0,0] #x,z, relative to centre of block
PipeDiam = 0.01 ###Inner Diameter
PipeThick = 0.001
PipeLength = BlockLength
TileCentre = [0,0]
TileWidth = BlockWidth
TileLength = 0.03 #y
TileHeight = 0.005 #z
# Main Mesh parameters
Length1D = 0.005
Length2D = 0.005
Length3D = 0.005
# Pipe Sub-Mesh Parameter (number of segments for pipe circumference)
CircDisc = 20
# Tile Sub-Mesh Parameter
Sub2_1D = 0.001

## ERMES solver is used to creat BC for induction heating
RunEM = 'N'
CoilType = 'HIVE'
CoilDisp = [0, 0, 0.002]
Frequency = 1e4
Current = 10 #Current in the coil

NProc = 2

## HTC between cooland and pipe
CreateHTC = 'N' 
PipeGeom = 'smooth tube'
FluidT = 20 #Celcius
FluidP = 1 #MPa
FluidV = 10 #m/s


############################
##### Code_aster study #####
############################
RunStudy = 'Y'

CommFile = 'Amaze_Adapt'
Model = '3D'
Solver = 'MUMPS'

### Materials
Materials = {'Block':'Copper_NL', 'Pipe':'Copper_NL', 'Tile':'Tungsten_NL'}

### IC - Need either an initial temperature or a results file to import
ImportRes = 'n'
InitTemp = 20 #Celcius

### Time-stepping and temporal discretisation
Theta = 0.5
dt = [(0.001,10)] #timestep size and number of steps
ResStore = 3 #How often should results be stored
CheckFirst = 3
CheckEvery = 3 #If this is smaller than ResStore then ResStore will be used



############################
###### Post-processing #####
############################
RunPostProc = 'n'
ParaVisFile = 'ParaVis'



