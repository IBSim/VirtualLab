from types import SimpleNamespace as Namespace

Sim = Namespace()
Sim.Name = ['Mechanical', 'Thermal','TransientThermal']
#############
## Pre-Sim ##
#############

#############
### Aster ###
#############

Sim.AsterFile = Sim.Name # The CodeAster command file can be found in Scripts/$SIMULATION/Aster

#############
## Post-Sim #
#############
