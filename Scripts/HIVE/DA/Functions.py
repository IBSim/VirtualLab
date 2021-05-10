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


def DataScale(data,const,scale):
    '''
    This function scales n-dim data to a specific range.
    data: N-darray or scalar
    const: N-darray or scalar
    scale: N-darray or scalar
    Examples:
     - Normalising data:
        const=mean, scale=stddev
     - [0,1] range:
        const=min, scale=max-min
    '''
    return (data - const)/scale

def DataRescale(data,const,scale):
    '''
    This function scales data back to original range.
    data: N-darray or scalar
    const: N-darray or scalar
    scale: N-darray or scalar
    '''
    return data*scale + const


class Sampling():
    def __init__(self, method, dim=0, range=[], bounds=True, options={}):
        # Must have either a range or dimension
        if range:
            self.range = range
            self.dim = len(range)
        elif dim:
            self.range = [(0,1)]*dim
            self.dim = dim
        else:
            print('Error: Must provide either dimension or range')

        self.options = options
        self.bounds = bounds
        self._boundscomplete = False
        self._Nb = 0

        if method.lower() == 'halton': self.sampler = self.Halton
        elif method.lower() == 'random': self.sampler = self.Random
        elif method.lower() == 'sobol': self.sampler = self.Sobol
        elif method.lower() == 'grid': self.sampler = self.Grid
        elif method.lower() == 'subspace': self.sampler = self.SubSpace


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

    def _SubSpace_dist(self, N, f, d):
        Dist = []
        for i in range(d):
            SS_N = int(np.ceil(N*f))
            Dist.append(SS_N)
            N -= SS_N
        Dist.append(N)
        return Dist

    def _SubSpace_fn(self,N):
        # This function has looked at empricial results and seems to match well
        # with curves for 2,3 and 4 D
        n = N**(1/self.dim) # self.dim th root of N
        return (1 - 1/n)**2

    def _SubSpace(self):
        # defaults which can be updated using options
        options = {'dim': self.dim - 1, # number of dimensions down to survey
                   'method': 'halton',
                   'roll' : True,
                   'SS_Func' : self._SubSpace_fn}
        options.update(self.options)

        # Setup
        d = options['dim']
        if not hasattr(self,'generator'):
            self.Add = 0
            self.SumNb = np.array([0]*(d+1))
            self.IndNb = np.array([0]*(d+1))
            self.generator = [[] for _ in range(self.dim)]

            self._generator,self.SS = [],[]
            if options['method'].lower() == 'halton':
                import ghalton
                for i in range(d+1):
                    self._generator.append(ghalton.GeneralizedHalton(self.dim,0))
                    self.SS.append(special.binom(self.dim,self.dim-i)*2**i)

            elif options['method'].lower() == 'sobol':
                import torch
                for i in range(d+1):
                    self._generator.append(torch.quasirandom.SobolEngine(dimension=self.dim))
                    self.SS.append(special.binom(self.dim,self.dim-i)*2**i)

        # work out how many points from each subspace we will need
        fact = options['SS_Func'](self.Add+self.N)
        fact = max(fact,0.1) # ensures fact is never zero
        Dist = self._SubSpace_dist(self.Add+self.N,fact,d)
        # Divide by number of subspaces for each dimension & work out difference
        Ind = np.ceil(np.array(Dist)/self.SS) - self.IndNb
        # Get the necessary points needed and add to 'generator'
        for i, (num, gen) in enumerate(zip(Ind,self._generator)):
            if num == 0: continue
            # get num points from generator
            if options['method'].lower() == 'halton':
                genvals = gen.get(int(num))
            elif options['method'].lower() == 'sobol':
                genvals = gen.draw(int(num)).detach().numpy().tolist()
            self.IndNb[i]+=num # keep track of how many of each dim we've asked for

            if i==0:
                points = genvals
            else:
                points = []
                BndIxs = list(combinations(list(range(self.dim)),i)) # n chose i
                BndPerms = list(product(*[[0,1]]*i))
                for genval,BndIx,BndPerm in product(genvals,BndIxs,BndPerms):
                    genval = np.array(genval)
                    # roll generator to put bases out of sync (optional)
                    if options['roll']:
                        genval = np.roll(genval,i)

                    genval[list(BndIx)] = list(BndPerm)
                    points.append(genval.tolist())

            self.generator[i].extend(points)

        # Get N points one at a time from self.generator
        # to ensures ordering is maintained
        points = []
        for i in range(1,self.N+1):
            _N = self.Add+i
            fact = options['SS_Func'](_N)
            fact = max(fact,0.1) # ensures fact is never zero

            Dist = self._SubSpace_dist(_N,fact,d)
            # Get index of next point
            ix = (np.array(Dist) - self.SumNb).argmax()
            point = self.generator[ix].pop(0)
            points.append(point)
            self.SumNb[ix]+=1

        self.Add+=self.N

        return points

    def SubSpace(self):
        Bnds = self.getbounds()
        if self.N == 0: return Bnds

        Points = self._SubSpace()

        if isinstance(Bnds,np.ndarray):
            return np.vstack((Bnds,Points))
        else :
            return Points
