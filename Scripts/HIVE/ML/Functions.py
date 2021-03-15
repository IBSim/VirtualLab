import numpy as np
from scipy import spatial
from itertools import product

from Scripts.Common.VLFunctions import MeshInfo
'''
Add in Uniformity 1 which looks at stdev
'''
def Uniformity2(JHNode, MeshFile):
    Meshcls = MeshInfo(MeshFile)
    CoilFace = Meshcls.GroupInfo('CoilFace')
    Area, JHArea = 0, 0 # Actual area and area of triangles with JH
    for nodes in CoilFace.Connect:
        vertices = Meshcls.GetNodeXYZ(nodes)
        # Heron's formula
        a, b, c = spatial.distance.pdist(vertices, metric="euclidean")
        s = 0.5 * (a + b + c)
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        Area += area

        vertices[:,2] += JHNode[nodes - 1].flatten()

        a, b, c = spatial.distance.pdist(vertices, metric="euclidean")
        s = 0.5 * (a + b + c)
        area1 = np.sqrt(s * (s - a) * (s - b) * (s - c))
        JHArea += area1
    Meshcls.Close()

    Uniformity = JHArea/Area
    return Uniformity


class Sampling():
    def __init__(self, method, dim=0, range=[], bounds=True):
        # Must have either a range or dimension
        if range:
            self.range = range
            self.dim = len(range)
        elif dim:
            self.range = [(0,1)]*dim
            self.dim = dim
        else:
            print('Must provide either dimension or range')

        self.bounds=bounds
        self._boundscomplete = False
        self._Nb = 0

        if method.lower() == 'halton': self.sampler = self.Halton
        elif method.lower() == 'random': self.sampler = self.Random
        elif method.lower() == 'sobol': self.sampler = self.Sobol
        elif method.lower() == 'grid': self.sampler = self.Grid


    def get(self,N):
        self.N=N
        norm = self.sampler()
        range = np.array(self.range)
        scale = norm*(range[:,1] - range[:,0]) + range[:,0]
        return scale.T.tolist()

    def getbounds(self):
        if self.bounds and not self._boundscomplete:
            Bnds = np.array(list(zip(*product(*[[0,1]]*self.dim)))).T
            Bnds = Bnds[self._Nb:self._Nb + self.N]
            self._Nb += Bnds.shape[0]
            if self._Nb < 2**self.dim:
                self.N=0
            else:
                self._boundscomplete=True
                self.N -= Bnds.shape[0] # number of N to ask for from Halton
            return Bnds

    def Halton(self):
        import ghalton
        if not hasattr(self,'generator'):
            self.generator = ghalton.Halton(self.dim)

        Bnds = self.getbounds()
        if self.N == 0: return Bnds

        Points = np.array(self.generator.get(self.N))

        if isinstance(Bnds,np.ndarray):
            return np.vstack((Bnds,Points))
        else :
            return Points

    def Random(self):
        Bnds = self.getbounds()
        if self.N == 0: return Bnds

        Points = np.random.uniform(size=(self.N,self.dim))
        if isinstance(Bnds,np.ndarray):
            return np.vstack((Bnds,Points))
        else :
            return Points

    def Sobol(self):
        import torch
        if not hasattr(self,'generator'):
            self.generator = torch.quasirandom.SobolEngine(dimension=self.dim)

        Bnds = self.getbounds()
        if self.N == 0: return Bnds

        Points = self.generator.draw(self.N).detach().numpy().tolist()
        if isinstance(Bnds,np.ndarray):
            return np.vstack((Bnds,Points))
        else :
            return Points


    def Grid(self):
        disc = np.linspace(0,1,self.N)
        return np.array(list(zip(*product(*[disc]*self.dim)))).T
