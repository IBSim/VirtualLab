from types import SimpleNamespace as Namespace

##########################
######## Meshing #########
##########################
Mesh = Namespace()
Mesh.Name = 'NoVoid'
Mesh.File = 'Disc'

# Geometric parameters
Mesh.Radius = 0.0063
# Height of bottom part of disk.
Mesh.HeightB = 0.00125
# Height of top part of disk.
Mesh.HeightT = 0.00125
# Optional parameters for void.
Mesh.VoidRadius = 0.0
# Height of Void. Positive/negative value creates void in top/bottom disk respectively.
Mesh.VoidHeight = 0.0
# Void centre relative to centre of disk ((0, 0) is at the centre).
Mesh.VoidCentre = (0,0)

# Parameters to generate mesh
Mesh.Length1D = 0.0003
Mesh.Length2D = 0.0003
Mesh.Length3D = 0.0003
# Number of segments for void circumference (if no void exists).
Mesh.VoidSegmentN = 30

##########################
####### Simulation #######
##########################
Sim = Namespace()
Sim.Name = 'Single'

#############
### Aster ###
#############
Sim.AsterFile = 'Disc_Lin'
Sim.Mesh = 'NoVoid'

# Modelisation & solver
Sim.Model = '3D'
Sim.Solver = 'MUMPS'

# Boundary Condtions
# Energy imparted by lasrer
Sim.Energy = 5.32468714
# Lasrer temporal profile (see Scripts/LFA/Laser for all options)
Sim.LaserT= 'Trim'
# Laser spatial profile (Gaussian or Uniform)
Sim.LaserS = 'Gauss'
# Heat transfer coefficient
Sim.BottomHTC = 0
Sim.TopHTC = 0
Sim.ExtTemp = 20

# Initial Conditions
Sim.InitTemp = 20

# Material type(s) for analysis, the properties of which can be found in the 'Materials' directory
Sim.Materials = {'Top':'Copper', 'Bottom':'Copper'}

# Time-stepping and temporal discretisation
Sim.dt = [(0.00002,50,1), (0.0005,100,2)]
Sim.Theta = 0.5

#############
# PostAster #
#############
Sim.PostAsterFile = 'ConductivityCalc'
Sim.Rvalues = [0.1, 0.5]
