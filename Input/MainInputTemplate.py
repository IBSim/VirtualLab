from types import SimpleNamespace as Namespace

##########################
######## Meshing #########
##########################
Mesh = Namespace()
Mesh.Name = 'NameOfMeshToCreate'
Mesh.File = 'FileToCreateMesh' # This file must be in Scripts/$SIMULATION/PreProc

##########################
####### Simulation #######
##########################
Sim = Namespace()
Sim.Name = 'NameOfSimulation'

#############
## PreAster #
#############

#############
### Aster ###
#############
Sim.CommFile = 'FileToRunSimulation' # This file must be in Scripts/$SIMULATION/Aster
Sim.Mesh = 'NameOfMeshUsedInSimulation' # The mesh used in the simulation

#############
# PostAster #
#############

