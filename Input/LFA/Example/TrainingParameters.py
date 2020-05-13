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
Mesh.HoleDisc = 30 # Number of segments for hole circumference (for sub-mesh)

##########################
####### Simulation #######
##########################
Sim = Namespace()
Sim.Name = 'Single'
#############
## Pre-Sim ##
#############

#############
### Aster ###
#############
Sim.CommFile = 'Disc_Lin'
Sim.Mesh = 'NoVoid'
Sim.Model = '3D'
Sim.Solver = 'MUMPS'
Sim.ResName = 'Thermal'
# Material type(s) for analysis, the properties of which can be found in the 'Materials' directory
Sim.Materials = {'Top':'Copper_NL', 'Bottom':'Copper_NL'}
# Initial Conditions - Need either an initial temperature or a results file to import
Sim.InitTemp = 20
Sim.ImportRes = 'n'
# Laser profile
Sim.Energy = 5.32468714
Sim.LaserT= 'Trim' #Temporal profile (see Scripts/LFA/Laser for all options)
Sim.LaserS = 'Gauss' #Spatial profile (Gauss profile or uniform profile available)
Sim.LaserEnd = 0.0008 # If this is not supplied the final time in the LaserT file will be used
# Boundary Condtions
Sim.ExtTemp = 20
Sim.BottomHTC = 0
Sim.TopHTC = 0
# Time-stepping and temporal discretisation
Sim.dt = [(0.00004,20), (0.0005,100)]
Sim.Theta = 0.5
Sim.Storing = {'Laser':1, 'PostLaser':2}

#############
## Post-Sim #
#############
Sim.PostCalcFile = 'DiscPost'
Sim.Rvalues = [0.1, 0.5, 1]
#Sim.ParaVisFile = 'DiscPV'

