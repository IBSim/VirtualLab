import os
import sys
import numpy as np
from scipy import spatial, special
import h5py
import pickle
from natsort import natsorted
from importlib import import_module, reload
from Scripts.Common.VLFunctions import MeshInfo

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

def MSE(Predicted,Target):
    sqdiff = (Predicted - Target)**2
    return np.mean(sqdiff)

def GetResPaths(ResDir,DirOnly=True,Skip=['_']):
    ''' This iterates over the directories in ResDir and runs fnc in each'''

    ResPaths = []
    for _dir in natsorted(os.listdir(ResDir)):
        if _dir.startswith(tuple(Skip)): continue
        path = "{}/{}".format(ResDir,_dir)
        if DirOnly and os.path.isdir(path): ResPaths.append(path)

    return ResPaths

def ReadData(datapkl):
    DataDict = {}
    with open(datapkl, 'rb') as fr:
        try:
            while True:
                pkldict = pickle.load(fr)
                DataDict = {**pkldict}
        except EOFError:
            pass
    return DataDict

def ReadParameters(paramfile):
    paramdir = os.path.dirname(paramfile)
    paramname = os.path.splitext(os.path.basename(paramfile))[0]
    sys.path.insert(0,paramdir)
    try:
        Parameters = reload(import_module(paramname))
    except ImportError:
        parampkl = "{}/.{}.pkl".format(paramdir,paramname)
        with open(parampkl,'rb') as f:
            Parameters = pickle.load(f)
    sys.path.pop(0)
    return Parameters

def Writehdf(File, array, dsetpath):
    Database = h5py.File(File,'a')
    if dsetpath in Database:
        del Database[dsetpath]
    Database.create_dataset(dsetpath,data=array)
    Database.close()
