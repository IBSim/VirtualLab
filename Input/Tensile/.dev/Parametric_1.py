from types import SimpleNamespace as Namespace

##########################
######## Meshing #########
##########################
Mesh = Namespace()
Mesh.Name = ['Notch2','Notch3','Notch4']
Mesh.Rad_a = [0.001,0.002,0.001]
Mesh.Rad_b = [0.001,0.0005,0.0005]
# Mesh.Run = [True,True,False]

##########################
####### Simulation #######
##########################
Sim = Namespace()
Sim.Name = ['Parametric1', 'Parametric2']
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
