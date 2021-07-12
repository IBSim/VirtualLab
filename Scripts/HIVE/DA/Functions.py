import numpy as np
from scipy import spatial, special
from itertools import product, combinations
from scipy.optimize import minimize

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

def Uniformity3(JHNode, MeshFile):
    Meshcls = MeshInfo(MeshFile)
    CoilFace = Meshcls.GroupInfo('CoilFace')

    Area, JHVal = 0, [] # Actual area and area of triangles with JH
    for nodes in CoilFace.Connect:
        vertices = Meshcls.GetNodeXYZ(nodes)
        # Heron's formula
        a, b, c = spatial.distance.pdist(vertices, metric="euclidean")
        s = 0.5 * (a + b + c)
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        Area += area

        v = np.mean(JHNode[nodes])

        JHVal.append(area*v)

        vertices[:,2] += JHNode[nodes - 1].flatten()

    Meshcls.Close()

    Variation = np.std(JHVal)
    return Variation

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

def _MinMax(X, fn, sign, *args):
    val,grad = fn(X,*args)
    return sign*val,sign*grad

def FuncOpt(fnc, NbInit, bounds, find='max',order='decreasing', tol=0.01, **kwargs):
    if find.lower()=='max':sign=-1
    elif find.lower()=='min':sign=1
    _Optima, fnVal, fnGrad, Coord = [],[],[],[]

    kwargs['args'] = (fnc,sign,*kwargs.get('args',[]))
    for X0 in np.random.uniform(0,1,size=(NbInit,4)):
        Opt = minimize(_MinMax, X0, jac=True, method='SLSQP', bounds=bounds,**kwargs)
        _Optima.append(Opt)
        if Opt.success:
            fnVal.append(Opt.fun)
            fnGrad.append(Opt.jac)
            Coord.append(Opt.x)

    fnVal,fnGrad,Coord = sign*np.array(fnVal),sign*np.array(fnGrad),np.array(Coord)
    #Sort Optimas in increasing/decreasing order
    ord = -1 if order.lower()=='decreasing' else 1
    sortIx = np.argsort(fnVal)[::ord]
    fnVal,fnGrad,Coord = fnVal[sortIx],fnGrad[sortIx],Coord[sortIx]

    if tol:
        tolCd,Ix = Coord[:1],[0]
        for i, cd in enumerate(Coord[1:]):
            D = np.linalg.norm(tolCd - cd, axis=1)
            if all(D>tol):
                Ix.append(i+1)
                tolCd = np.vstack((tolCd,cd))

        fnVal,fnGrad,Coord = fnVal[Ix],fnGrad[Ix],Coord[Ix]

    return Coord, fnVal, fnGrad

def MSE(Predicted,Target):
    sqdiff = (Predicted - Target)**2
    return np.mean(sqdiff)
