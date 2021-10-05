from types import SimpleNamespace as Namespace
import numpy as np
from Scripts.Common.VLFunctions import Sampling

'''
This file enables different sampling methods to be used for collecting data for
simualtions. The options are: Grid, Random, Halton or Adaptive.
Adaptive is ina  sepearte runs cript due to the continuous feedback requirements
'''
Sim = Namespace()

DispX = [-0.02,0.02]
DispY = [-0.02,0.02]
DispZ = [0.0015,0.005]
Rotation = [-5,5]
bounds = [DispX,DispY,DispZ,Rotation]

Method = 'Halton'
N = 30

Sampler = Sampling(Method,range=bounds)
Samples = Sampler.get(N)
Sim.CoilDisplacement = np.vstack((Samples[:3])).T.tolist()
Sim.Rotation = Samples[3]
Sim.Name = ["{}/Sim_{}".format(Method,i) for i in range(N)]
