import os
import sys

import h5py
import numpy as np

from Scipts.Common.tools import MeshInfo

def Variation(ResFile,ResName='Joule_heating',Type='grad'):
    g = h5py.File(ResFile, 'r')
    gRes = g['/CHA/{}'.format(ResName)]
    step = list(gRes.keys())[0]
    JH_Node = gRes["{}/NOE/MED_NO_PROFILE_INTERNAL/CO".format(step)][:]
    g.close()

    meshdata = MeshInfo(ResFile)
    CoilFace = meshdata.GroupInfo('CoilFace')
    Connect = CoilFace.Connect
    NodeIDs = list(range(1,meshdata.NbNodes+1))
    Coor = meshdata.GetNodeXYZ(NodeIDs)
    meshdata.Close()

    _Ix = np.searchsorted(NodeIDs,Connect)
    elem_cd = Coor[_Ix]
    v1,v2 = elem_cd[:,1] - elem_cd[:,0], elem_cd[:,2] - elem_cd[:,0]
    cross = np.cross(v1,v2)
    area = 0.5*np.linalg.norm(cross,axis=1)

    _jh = JH_Node[_Ix]/10**6

    if Type.lower() == 'grad':
        elem_cd[:,:,2] = _jh
        v1,v2 = elem_cd[:,1] - elem_cd[:,0], elem_cd[:,2] - elem_cd[:,0]
        cross = np.cross(v1,v2)
        crossxy = cross[:,:2]/cross[:,2:3]
        crossmag = np.linalg.norm(crossxy,axis=1)
        Var = (area*crossmag).sum()

    elif Type.lower() == 'std':
        # This method will be poor if mesh fineness is variable
        JH_avg = _jh.mean(axis=1)
        Var = np.std(JH_avg)

    return Var

def Power(ResFile,ResName='Joule_heating'):
    g = h5py.File(ResFile, 'r')
    gRes = g['/CHA/{}'.format(ResName)]
    step = list(gRes.keys())[0]
    JH_Node = gRes["{}/NOE/MED_NO_PROFILE_INTERNAL/CO".format(step)][:]
    g.close()

    meshdata = MeshInfo(ResFile)

    NodeIDs = list(range(1,meshdata.NbNodes+1))
    Coor = meshdata.GetNodeXYZ(NodeIDs)
    Sample = meshdata.GroupInfo('Sample')
    Connect = Sample.Connect

    _Ix = np.searchsorted(NodeIDs,Connect)

    # work out volume of each element
    elem_cd = Coor[_Ix]
    v1,v2 = elem_cd[:,1] - elem_cd[:,0], elem_cd[:,2] - elem_cd[:,0]
    v3 = elem_cd[:,3] - elem_cd[:,0]
    cross = np.cross(v1,v2)
    dot = (cross*v3).sum(axis=1)
    Volumes = 1/float(6)*np.abs(dot)

    # work out average joule heating per volume
    _jh = JH_Node[_Ix]
    JH_vol = _jh.mean(axis=1)

    # Calculate power
    P = (Volumes*JH_vol).sum()

    return P

def Single(VL,SimDict):
    Parameters = SimDict["Parameters"]
    ResFile = '{}/ERMES.rmed'.format(SimDict['PREASTER'])

    P = Power(ResFile)
    SimDict['Data']['Power'] = P

    V = Variation(ResFile)
    SimDict['Data']['Variation'] = V

    print(P)
    print(V)
