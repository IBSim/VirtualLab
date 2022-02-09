import h5py
import numpy as np
import os
import sys
from VLFunctions import MeshInfo

def TC_Temperature(ResFile,TCLocations,ResName='Temperature',Collect='nearest'):

#============================================================================
    # Get mesh information from the results file
    meshdata = MeshInfo(ResFile)
    SurfaceNormals = np.array([['TileFront', 'NX'], ['TileBack', 'NX'], ['TileSideA', 'NY'],
                              ['TileSideB', 'NY'], ['TileTop', 'NZ'],
                              ['BlockFront', 'NX'], ['BlockBack', 'NX'], ['BlockSideA', 'NY'],
                              ['BlockSideB', 'NY'],['BlockBottom', 'NZ'], ['BlockTop', 'NZ']])

    a = np.array(TCLocations)
    SurfaceNames = np.char.add(*a[:,:2].T) # combine entries for part and face

    TC_locs = a[:,2:].astype('float64')
    GetNode,Weights = [None]*len(SurfaceNames),[1]*len(SurfaceNames)
    Collect = 'interpolated'
    # Collect='nearest'
    for SurfaceName in np.unique(SurfaceNames):
        # print(SurfaceName)
        Ixs = (SurfaceNames==SurfaceName).nonzero()[0]
        if len(Ixs)==0:
            print('error')
            continue

        GroupInfo = meshdata.GroupInfo(SurfaceName)

        NodeIDs = GroupInfo.Nodes
        Connect = GroupInfo.Connect
        ElemIDs = GroupInfo.Elements

        Coords = meshdata.GetNodeXYZ(NodeIDs)

        c_min = Coords.min(axis=0)
        c_max = Coords.max(axis=0)
        range = c_max - c_min

        norm = SurfaceNormals[SurfaceNormals[:,0]==SurfaceName,1]
        if norm == 'NX': fix,chng = [0],[1,2]
        elif norm == 'NY': fix,chng = [1],[0,2]
        elif norm == 'NZ': fix,chng = [2],[0,1]
        TC_loc = np.zeros(3)
        TC_loc[fix] = c_min[fix]

        for Ix in Ixs:
            # print(Ix)
            TC_loc[chng] = c_min[chng] + range[chng]*TC_locs[Ix].flatten()
            # print(TC_loc)

            if Collect.lower()=='nearest':
                d = np.linalg.norm(Coords - TC_loc,axis=1)
                cl_ix = np.argmin(d)
                GetNode[Ix] = [NodeIDs[cl_ix]]
            elif Collect.lower()=='interpolated':
                # Get coordinates for each node in the connectivity
                _Ix = np.searchsorted(NodeIDs,Connect)
                # _Ix = np.searchsorted(NodeIDs,Connect,sorter=NodeIDs.argsort())
                a = Coords[_Ix]

                # Use only changing values across surface
                a1,a2 = a[:,:,chng[0]],a[:,:,chng[1]]
                # c = np.stack((a1,a2,np.ones(a1.shape)),axis=1)
                # Areas = 0.5*np.linalg.det(c)
                # print(Areas.sum())
                biareas = []
                for ls in [[1,2],[2,0],[0,1]]:
                    _a1 = a1[:,ls]
                    _a2 = a2[:,ls]
                    _d = np.ones((len(_a1),1))
                    _a1 = np.concatenate((_a1,_d*TC_loc[chng[0]]),axis=1)
                    _a2 = np.concatenate((_a2,_d*TC_loc[chng[1]]),axis=1)
                    _c = np.stack((_a1,_a2,np.ones(_a1.shape)),axis=1)
                    _area = 0.5*np.linalg.det(_c)
                    biareas.append(_area)
                biareas = np.array(biareas).T
                # TC_loc is member of element where all bi areas have the same sign
                # sum_sign is 3 when this happens
                sum_sign = np.abs(np.sign(biareas).sum(axis=1))
                elemix = (sum_sign==3).nonzero()[0]

                # get weighting for each contribution
                biarea = biareas[elemix]
                weighting = biarea/biarea.sum()

                GetNode[Ix] = Connect[elemix].flatten()
                Weights[Ix] = weighting.flatten()
                if False:
                    biarea_sum = biarea.sum()
                    c = np.stack((a1[elemix],a2[elemix],np.ones((1,3))),axis=1)
                    area = 0.5*np.linalg.det(c)[0]
                    print(biarea_sum,' = ',area)
                    # c = np.stack((a1,a2,np.ones(a1.shape)),axis=1)
                    # Areas = 0.5*np.linalg.det(c)


    GetNode = np.array(GetNode)-1
    Weights = np.array(Weights).reshape(GetNode.shape)
    # print(GetNode.shape)
#============================================================================
    # open result file using h5py
    g = h5py.File(ResFile, 'r')
    gRes = g['/CHA/{}'.format(ResName)]
    steps = gRes.keys()

    Times, Temperatures = [], []
    for step in steps:
        Times.append(gRes[step].attrs['PDT'])
        Temp = gRes['{}/NOE/MED_NO_PROFILE_INTERNAL/CO'.format(step)][:]
        TC_Temps = Temp[GetNode]
        TC_Temps = (TC_Temps*Weights).sum(axis=1)
        Temperatures.append(TC_Temps)

    g.close()

    Temperatures = np.array(Temperatures)

    return Temperatures, Times

def MaxTemperature(ResFile,ResName='Temperature'):
    # open result file using h5py
    g = h5py.File(ResFile, 'r')
    gRes = g['/CHA/{}'.format(ResName)]
    steps = gRes.keys()

    Temperatures,Times = [],[]
    for step in steps:
        Temp = gRes["{}/NOE/MED_NO_PROFILE_INTERNAL/CO".format(step)][:]
        Temperatures.append(Temp.max())
        Times.append(gRes[step].attrs['PDT'])

    g.close()

    Temperatures = np.array(Temperatures)
    return Temperatures,Times

def MaxStress(ResFile,ResName='Stress'):
    g = h5py.File(ResFile, 'r')
    gRes = g['/CHA/Stress/']
    step = list(gRes.keys())[0]
    Stress = gRes['{}/MAI.TE4/MED_NO_PROFILE_INTERNAL/CO'.format(step)][:]

    g.close()

    Stress = Stress.reshape((int(Stress.size/6),6),order='F')
    Stress_mag = np.linalg.norm(Stress, axis=1)
    MaxStress = Stress_mag.max()

    return MaxStress
