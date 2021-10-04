from types import SimpleNamespace as Namespace

##########################
####### Simulation #######
##########################
Sim = Namespace()

# Name under which the simulation results will be saved.
Sim.Name = 'IBSim'
# The CodeAster command file can be found in Scripts/$SIMULATION.
Sim.AsterFile = 'Tensile'

# The mesh used in the simulation.
Sim.Mesh = 'Tensile_IBSim'
# Enforced displacement in displacement controlled analysis.
Sim.Displacement = 0.01
# Material specimen is made of. Properties can be found in the 'Materials' directory.
Sim.Materials = 'Copper'
