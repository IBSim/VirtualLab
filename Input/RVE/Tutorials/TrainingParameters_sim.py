from types import SimpleNamespace as Namespace

Sim = Namespace()


Sim.Name = 'k'
Sim.AsterFile = 'py3'
Sim.Mesh = 'Notch2'
Sim.dpa=1
Sim.temp_gradientx=.38
Sim.temp_gradienty=.38
Sim.temp_gradientz=.38
Sim.temp=2
Sim.condTungsten=7
Sim.condRhenium=7
Sim.condOsmium=7
Sim.Pipe = {'Type':'smooth tube', 'Diameter':0.012, 'Length':0.012}
Sim.Coolant ={'Temperature':150, 'Pressure':5, 'Velocity':10}

