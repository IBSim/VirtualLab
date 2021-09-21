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
Sim.Name = ['Linear/SimNoVoid','Linear/SimVoid1','Linear/SimVoid2']

Sim.Mesh = ['NoVoid','Void', 'Void']
Sim.LaserS = ['Uniform', 'Gauss', 'Uniform']
Sim.Run = [True, False, False]
