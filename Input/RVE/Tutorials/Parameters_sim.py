from types import SimpleNamespace as Namespace
import numpy as N
##########################
######## Meshing #########
##########################

##########################
####### Simulation #######
##########################
#
Sim = Namespace()
dpa=[1]
e=len(dpa)
name=[]
for i in range(0,e):
    name.append('microstructure'+str(i))

Sim.Name = name
Sim.AsterFile = ['RVE']
Sim.Mesh = ['RVE']
Sim.dpa=[1]
Sim.temp_gradientx=[.38]
Sim.temp_gradienty=[.38]
Sim.temp_gradientz=[.38]
Sim.temp=[200]
Sim.condTungsten=[.17]
Sim.condRhenium=[.039]
Sim.condOsmium=[.075]
Sim.Pipe = [{'Type':'smooth tube', 'Diameter':0.012, 'Length':0.012}]
Sim.Coolant =[{'Temperature':150, 'Pressure':5, 'Velocity':10}]

