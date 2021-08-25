from types import SimpleNamespace as Namespace

##########################
######## Meshing #########
##########################
Mesh = Namespace()

Mesh.Name = ['MeshCase1', 'MeshCase2', 'MeshCase3', 'MeshCase4', 'MeshCase5', 'MeshCase6', 'MeshCase7', 'MeshCase8', 'S12Case9' ]

Mesh.Void = [] # void dimensions and orientation
Mesh.VoidCentre = [] # void location

Mesh.Void.append([])
Mesh.Void.append([])
Mesh.Void.append([[	0.0075	,	0.0075	,	0.005	,	0.00	]])
Mesh.Void.append([[	0.0075	,	0.0075	,	0.005	,	0.00	]])
Mesh.Void.append([[	0.00375	,	0.00375	,	0.005	,	0.000	], [	0.00375	,	0.00375	,	0.005	,	0.000	], [	0.00375	,	0.00375	,	0.005	,	0.000	], [	0.00375	,	0.00375	,	0.005	,	0.000	]])
Mesh.Void.append([[	0.005	,	0.01125	,	0.005	,	0.00	]]) 
Mesh.Void.append([[	0.01125	,	0.005	,	0.005	,	0.00	]])																								
Mesh.Void.append([[	0.00625	,	0.00225	,	0.005	,	0.00	], [	0.00625	,	0.00225	,	0.005	,	0.00	], [	0.00625	,	0.00225	,	0.005	,	0.00	], [	0.00625	,	0.00225	,	0.005	,	0.00	]])
Mesh.Void.append([[	0.00225	,	0.00625	,	0.005	,	0.00	], [	0.00225	,	0.00625	,	0.005	,	0.00	], [	0.00225	,	0.00625	,	0.005	,	0.00	], [	0.00225	,	0.00625	,	0.005	,	0.00	]])

Mesh.VoidCentre.append([])
Mesh.VoidCentre.append([])
Mesh.VoidCentre.append([[	0.5	,	0.5	]]) 
Mesh.VoidCentre.append([[	0.277778	,	0.277778	]]) 
Mesh.VoidCentre.append([[	0.250	,	0.250	], [	0.750	,	0.250	], [	0.250	,	0.750	], [	0.750	,	0.750	]])
Mesh.VoidCentre.append([[	0.277778	,	0.388889	]])  
Mesh.VoidCentre.append([[	0.388889	,	0.277778	]])		
Mesh.VoidCentre.append([[	0.277778	,	0.138889	], [	0.277778	,	0.305556	], [	0.277778	,	0.472222	], [	0.277778	,	0.638889	]])
Mesh.VoidCentre.append([[	0.138889	,	0.277778	], [	0.305556	,	0.277778	], [	0.472222	,	0.277778	], [	0.638889	,	0.277778	]])



##########################
####### Simulation #######
##########################
Sim = Namespace()
Sim.Name = ['SimulationCase1', 'SimulationCase2', 'SimulationCase3', 'SimulationCase4', 'SimulationCase5', 'SimulationCase6', 'SimulationCase7', 'SimulationCase8', 'SimulationCase9']

#############
## Pre-Sim ##
#############

#############
### Aster ###
#############

Sim.Mesh = Mesh.Name

Sim.TileBlockTCC = [None, 2.0e+4, 2.0e+4, 2.0e+4, 2.0e+4, 2.0e+4, 2.0e+4, 2.0e+4, 2.0e+4]
#############
## Post-Sim #
#############