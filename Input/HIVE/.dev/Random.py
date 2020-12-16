from types import SimpleNamespace as Namespace
import numpy as np

Rotation = [-5,5]
DispX = DispY = [-0.02,0.02]
DispZ = [0.0015,0.005]
Rotation = [-5,5]

Coil = 'Test'

Sim = Namespace()

St, Num = 0,3

Sim.Name = []
Sim.CoilDisplacement = []
Sim.Rotation = []
Sim.CoilType = [Coil]*Num
for i in range(St,St+Num):

    Disp = [np.random.uniform(*DispX),np.random.uniform(*DispY),np.random.uniform(*DispZ)]
    Rot = np.random.uniform(*Rotation)

    Sim.CoilDisplacement.append(Disp)
    Sim.Rotation.append(Rot)

    Sim.Name.append('{}_{}'.format(Coil,i))
