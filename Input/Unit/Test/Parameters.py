from types import SimpleNamespace as Namespace

Mesh = Namespace()
Mesh.Name = 'MeshCube'
Mesh.File = 'UnitCube'

Sim = Namespace()
Sim.Name = 'Single'
#############
## Pre-Sim ##
#############

#############
### Aster ###
#############

Sim.AsterFile = 'Mechanical' # The CodeAster command file can be found in Scripts/$SIMULATION/Aster
Sim.Mesh = 'MeshCube' # The mesh used in the simulation

#############
## Post-Sim #
#############
