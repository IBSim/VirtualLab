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
Sim.Name = ['SimNoVoid','SimVoid']
#############
## Pre-Sim ##
#############

#############
### Aster ###
#############
Sim.Mesh = ['NoVoid','Void']

#############
## Post-Sim #
#############


