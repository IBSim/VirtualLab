import numpy as np
from scipy import spatial, special
from itertools import product, combinations


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
        elif method.lower() == 'lewis': self.sampler = self.Lewis


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
                self.N -= Bnds.shape[0]
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

    def __Lewis(self):
        if not hasattr(self,'Add'):
            self.Add=0
            self.SumNb = np.array([0]*(self.dim))
            self.IndNb = np.array([0]*(self.dim))
            self.Generator = [[] for _ in range(self.dim)]

            import ghalton
            self._generator,self.SS = [],[]
            for i in range(self.dim):
                self._generator.append(ghalton.Halton(self.dim - i))
                self.SS.append(special.binom(self.dim,self.dim-i)*2**i)

        self.Add += self.N

        _N = self.Add


        n = _N**(1/self.dim)
        fact = (1 - 1/n)**2
        # fact = (1 - 1/n)**self.dim # Seems to be too affected by higher powers
        fact = max(fact,0.1) # ensures fact is never zero
        # print(fact)
        Dist = []
        for i in range(self.dim-1):
            Top = int(np.ceil(_N*fact))
            Dist.append(Top)
            _N -= Top
        Dist.append(_N)
        Dist = np.array(Dist)
        # print(Dist)
        Ind = np.ceil(Dist/self.SS) - self.IndNb

        for i, (num, gen, ss) in enumerate(zip(Ind,self._generator,self.SS)):
            if num == 0: continue
            genvals = gen.get(int(num))
            self.IndNb[i]+=num # keep track of how many of each dim we've asked for
            if i==0:
                vals = genvals
            else:
                vals = []
                BndIxs = list(combinations(list(range(self.dim)),i)) # n chose i
                BndPerms = list(product(*[[0,1]]*i))
                for genval,BndIx,BndPerm in product(genvals,BndIxs,BndPerms):
                    IntIx = list(set(range(self.dim)).difference(BndIx))
                    a = np.zeros(self.dim)
                    a[IntIx] = genval
                    a[list(BndIx)] = list(BndPerm)
                    vals.append(a.tolist())

            self.Generator[i].extend(vals)

        # ensures ordering is maintained
        M = []
        oldNb = self.Add - self.N
        for i in range(1,self.N+1):
            _N = oldNb+i
            n = _N**(1/self.dim)
            fact = (1 - 1/n)**2
            fact = max(fact,0.1) # ensures fact is never zero

            Dist = []
            for i in range(self.dim-1):
                Top = int(np.ceil(_N*fact))
                Dist.append(Top)
                _N -= Top
            Dist.append(_N)
            Dist = np.array(Dist)
            Need = Dist - self.SumNb

            ix = Need.argmax()
            v = self.Generator[ix].pop(0)
            M.append(v)
            Need[ix]-=1
            self.SumNb[ix]+=1
        # print(M)
        return M

    def Lewis(self):
        Bnds = self.getbounds()
        if self.N == 0: return Bnds

        Points = self.__Lewis()

        if isinstance(Bnds,np.ndarray):
            return np.vstack((Bnds,Points))
        else :
            return Points


    def _Lewis(self):
        # alot of for loops here which could be tidied up
        if not hasattr(self,'a'): self.a = 1
        if not hasattr(self,'nbs'): self.nbs = [0]*self.dim
        dims = list(range(1,self.dim+1))[::-1]
        disc = [self.a**d for d in dims]

        import ghalton
        if not hasattr(self,'_generator'):
            self._generator = {}
            for i in dims:
                self._generator[i] = ghalton.Halton(i)

        M = []
        for i, (_dim,_disc) in enumerate(zip(dims,disc)):
            BIxs = list(combinations(dims,self.dim - _dim))
            BoundPerm = list(product(*[[0,1]]*(self.dim - _dim)))

            if True:
                _n = _disc - self.nbs[i]
                self.nbs[i] = _disc
            else:
                _n = _disc

            points = self._generator[_dim].get(_n)
            for point in points:
                for BIx in BIxs:
                    # print(point,BIx)
                    NBIx = set(dims) - set(BIx)
                    for cmb in BoundPerm:
                        ls = [0]*self.dim
                        for _NBIx,_p in zip(NBIx,point):
                            ls[_NBIx-1] = _p
                        for _BIx,_p in zip(BIx,cmb):
                            ls[_BIx-1] = _p

                        M.append(ls)

        self.a+=1

        self.generator = np.array(M)
        # print(self.generator)
    def LewisOld(self):
        Bnds = self.getbounds()
        if self.N == 0: return Bnds

        if not hasattr(self,'generator'): self._Lewis()

        if self.N < self.generator.shape[0]:
            Points = self.generator[:self.N,:]
            self.generator = self.generator[self.N:,]
        else:
            Points = self.generator
            need = self.N - Points.shape[0]
            while need>0:
                self._Lewis()
                if need >= self.generator.shape[0]:
                    NewPoints = self.generator
                else:
                    NewPoints = self.generator[:need,:]
                    self.generator = self.generator[need:,]

                Points = np.vstack((Points,NewPoints))
                need -=NewPoints.shape[0]

        if isinstance(Bnds,np.ndarray):
            return np.vstack((Bnds,Points))
        else :
            return Points







        #
        # self._it=0
        # while self.N:
        #     for dim,coef in zip(dims[self._it:],coef[self._it:]):
