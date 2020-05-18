from types import SimpleNamespace as Namespace

##########################
######## Meshing #########
##########################
Mesh = Namespace()
Mesh.Name = ['Notch2','Notch3']
Mesh.Rad_a = [0.001,0.002]
Mesh.Rad_b = [0.001,0.0005]

##########################
####### Simulation #######
##########################
Sim = Namespace()
Sim.Name = ['ParametricSim1', 'ParametricSim2']
#############
## Pre-Sim ##
#############

#############
### Aster ###
#############
Sim.Mesh = ['Notch2', 'Notch3']

#############
## Post-Sim #
#############


