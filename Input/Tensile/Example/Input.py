##########################
##### Pre-processing #####
##########################

# Mesh used for analysis should be input at 'MeshName'. This mesh should eiether be in the 'Mesh' directory or 'CreateMesh' flag should be set to 'Y'
MeshName = 'Notch1'
CreateMesh = 'Y'

# If 'CreateMesh' is 'Y' then a mesh will be created using the below parameters
# The Salome file to create the mesh can be found in Scripts/(Simulation)/PreProc
MeshFile = 'Testpiece'

# Measurements to create geometry, all dimensions are measured in metres
Thickness = 0.003
HandleWidth = 0.024
HandleLength = 0.024
GaugeWidth = 0.012
GaugeLength = 0.04
TransRad = 0.012
HoleCentre = (0.0,0.0)
Rad_a = 0.001
Rad_b = 0.0005

# Parameters to generate mesh
Length1D = 0.001
Length2D = 0.001
Length3D = 0.001
HoleDisc = 30 # Number of segments for hole circumference (for sub-mesh)


############################
##### Code_aster study #####
############################
RunStudy = 'Y'

# The CodeAster command file which can be found in Scripts/(Simulation)/Aster
CommFile = 'Tensile'
# Name of results file(s) which will be output by CodeAster
ResName = ['ConstForce', 'ConstDisp']

# The tensile command file you can run a constant force and/or constant displacement tensile test
# Loading type should 'Force' for constant force, 'Displacement'/'Disp' for constant displacement or 'All'/'Both' to run both analysis
LoadingType = 'All' 
# Magnitude of force for constant force analysis, measured in N
Force = 1000000 
# Magnitude of displacement for constant displacement analysis, measured in metres
Disp = 0.01 

# Material type(s) for analysis, the properties of which can be found in the 'Materials' directory
Materials = 'Copper' 


############################
###### Post-processing #####
############################



