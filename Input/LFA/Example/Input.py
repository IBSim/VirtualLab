##########################
##### Pre-processing #####
##########################

# Mesh used for analysis should be input at 'MeshName'. This mesh should eiether be in the 'Mesh' directory or 'CreateMesh' flag should be set to 'Y'
MeshName = 'NoVoid'
CreateMesh = 'N'

# If 'CreateMesh' is 'Y' then a mesh will be created using the below parameters
# The Salome file to create the mesh can be found in Scripts/(Simulation)/PreProc
MeshFile = 'Disc'
### Geometric parameters (all units in metres)
Radius = 0.0063 # Radius of disk
HeightB = 0.00125 # Height of bottom part of disk
HeightT = 0.00125 # Height of top part of disk
VoidCentre = (0,0) # Void centre relative to centre of disk - (0, 0) is at the centre
VoidRadius = 0.000 # Radius of void
VoidHeight = 0.0000 # Height of Void. Positive/negative number gives a void in the top/bottom disk respectively

### Mesh Parameters
Length1D = 0.0005
Length2D = 0.0005  
Length3D = 0.0005 
HoleDisc = 20 # Number of segments for hole circumference (for sub-mesh)

############################
##### Code_aster study #####
############################
RunStudy = 'N'

CommFile = 'Disc_Lin'
ResName = 'ResTher'

Model = '3D'
Solver = 'MUMPS'

### Materials
Materials = {'Top':'Copper_NL', 'Bottom':'Copper_NL'}

### IC - Need either an initial temperature or a results file to import
ImportRes = 'n'
InitTemp = 20

### Laser
Energy = 5.32468714
LaserT= 'Trim' #Temporal laser profile (see Scripts/LFA/Laser for all options)
LaserS = 'Gauss' #Spatial laser profile (Gauss profile or uniform profile available)

### Other BC
ExtTemp = 20
# HTC depends on the surface finish of the material 
BottomHTC = 0
TopHTC = 0

### Time-stepping and temporal discretisation
dt = [(0.00004,20), (0.0005,100)]

Theta = 0.5
Store1 = 1 ### When Laser is on
Store2 = 2 ### When Laser is off


############################
###### Post-processing #####
############################
RunPostProc = 'Y'

PostCalcFile = 'DiscPost'
Rvalues = [0.1, 0.5, 1]

#ParaVisFile = 




