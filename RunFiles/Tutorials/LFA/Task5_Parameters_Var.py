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
Sim.Name = ['SimNoVoid_NL','SimVoid1_NL','SimVoid2_NL']
#############
## Pre-Sim ##
#############

#############
### Aster ###
#############
Sim.Mesh = ['NoVoid','Void', 'Void']
Sim.LaserS = ['Uniform', 'Gauss', 'Uniform']

#############
## Post-Sim #
#############


