from types import SimpleNamespace as Namespace

##########################
######## Meshing #########
##########################
Mesh = Namespace()
Mesh.Name = ['NoVoid','Void']
Mesh.VoidRadius = [0, 0.001]
Mesh.VoidHeight = [0, 0.0003]

##########################
####### Simulation #######
##########################
Sim = Namespace()
Sim.Name = ['SimNoVoid','SimVoid1','SimVoid2']
#############
## Pre-Sim ##
#############

#############
### Aster ###
#############
Sim.Mesh = ['NoVoid','Void', 'Void']
Sim.LaserS = ['Gauss', 'Gauss', 'Uniform']

#############
## Post-Sim #
#############


