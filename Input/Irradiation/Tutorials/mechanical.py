from types import SimpleNamespace as Namespace

# Inputs for finite element mechanical simulation

Sim = Namespace()
Sim.Name = 'unirradiated_day0'

# HTC between coolant and pipe (need Coolant and Pipe properties)

Sim.Pipe = {'Type':'smooth tube', 'Diameter':0.012, 'Length':0.012}
Sim.Coolant = {'Temperature':150, 'Pressure':5, 'Velocity':10}
Sim.AsterFile ='mechanical'
Sim.Mesh ='mono'
Sim.dpa=0
Sim.hd=0
