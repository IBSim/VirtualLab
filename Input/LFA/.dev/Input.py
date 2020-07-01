from types import SimpleNamespace as Namespace

##########################
######## Meshing #########
##########################
Mesh = Namespace()
Mesh.Name = 'NoVoid'
Mesh.File = 'Disc'
# Geometric parameters (all units in metres)
Mesh.Radius = 0.0063 # Radius of disk
Mesh.HeightB = 0.00125 # Height of bottom part of disk
Mesh.HeightT = 0.00125 # Height of top part of disk
Mesh.VoidCentre = (0,0) # Void centre relative to centre of disk - (0, 0) is at the centre
Mesh.VoidRadius = 0.000 # Radius of void
Mesh.VoidHeight = 0.0000 # Height of Void. Positive/negative number gives a void in the top/bottom disk respectively
# Parameters to generate mesh
Mesh.Length1D = 0.0003
Mesh.Length2D = 0.0003
Mesh.Length3D = 0.0003
Mesh.VoidDisc = 30 # Number of segments for hole circumference (for sub-mesh)

##########################
####### Simulation #######
##########################
Sim = Namespace()
Sim.Name = 'Single'
#############
## PreAster #
#############

#############
### Aster ###
#############
Sim.AsterFile = 'Disc_Lin'
Sim.Mesh = 'NoVoid'
Sim.Model = '3D'
Sim.Solver = 'MUMPS'
Sim.ResName = 'Thermal'
# Material type(s) for analysis, the properties of which can be found in the 'Materials' directory
Sim.Materials = {'Top':'Copper_NL', 'Bottom':'Copper_NL'}
# Initial Conditions
Sim.InitTemp = 20
# Laser profile
Sim.Energy = 5.32468714
Sim.LaserT= 'Trim' #Temporal profile (see Scripts/LFA/Laser for all options)
Sim.LaserS = 'Gauss' #Spatial profile (Gauss profile or uniform profile available)
# Boundary Condtions
Sim.ExtTemp = 20
Sim.BottomHTC = 0
Sim.TopHTC = 0
# Time-stepping and temporal discretisation
Sim.dt = [(0.00004,20,1), (0.0005,20,1)]
Sim.Theta = 0.5

#############
# PostAster #
#############
Sim.PostAsterFile = 'DiscPost'
Sim.Rvalues = [0.1, 0.5, 1]
Sim.CaptureTime = 0.0108
