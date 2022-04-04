import pickle

import h5py
import numpy as np

import AsterPostProc as AsterPP
import ERMESPostProc as ERMESPP
from Scripts.Common.tools import MeshInfo

def Single(VL,SimDict):
    Parameters = SimDict["Parameters"]
    ResFile = '{}/Thermal.rmed'.format(SimDict['ASTER'])

    MaxTemp =  AsterPP.MaxTemperature(ResFile)[0][0]
    SimDict['Data']['MaxTemp'] = MaxTemp

    if hasattr(Parameters,'ThermoCouple'):
        TC_Temp = AsterPP.TC_Temperature(ResFile,Parameters.ThermoCouple)[0]
        SimDict['Data']['TC_Temp'] = TC_Temp

    _MaxStress = getattr(Parameters,'MaxStress',False)
    if _MaxStress:
        MaxStress = AsterPP.MaxStress(ResFile)
        MaxStress = MaxStress/10**6 # MPa
        SimDict['Data']['MaxStress'] = MaxStress

def Surface_Temperatures(VL,SimDict):
    Parameters = SimDict["Parameters"]
    ResFile = '{}/Thermal.rmed'.format(SimDict['ASTER'])

    g = h5py.File(ResFile, 'r')
    gRes = g['/CHA/{}'.format('Temperature')]
    step = list(gRes.keys())[0]
    Temps = gRes['{}/NOE/MED_NO_PROFILE_INTERNAL/CO'.format(step)][:]
    g.close()

    SurfaceNormals = np.array([['TileFront', 'NX'], ['TileBack', 'NX'], ['TileSideA', 'NY'],
                              ['TileSideB', 'NY'], ['TileTop', 'NZ'],
                              ['BlockFront', 'NX'], ['BlockBack', 'NX'], ['BlockSideA', 'NY'],
                              ['BlockSideB', 'NY'],['BlockBottom', 'NZ'], ['BlockTop', 'NZ']])

    meshdata = MeshInfo(ResFile)
    mesh_surface = meshdata.GroupNames()
    Data = {}
    for surface,normal in SurfaceNormals:
        if surface not in mesh_surface: continue
        GroupInfo = meshdata.GroupInfo(surface)
        NodeIDs = GroupInfo.Nodes
        temperatures = Temps[NodeIDs-1]
        Data[surface] = temperatures

        # Coords = meshdata.GetNodeXYZ(NodeIDs)
        # if normal == 'NX': Coords = Coords[:,[1,2]]
        # elif normal == 'NY': Coords = Coords[:,[0,2]]
        # elif normal == 'NZ': Coords = Coords[:,[0,1]]
        #
        # dat = np.concatenate((Coords,temperatures[:,None]),axis=1)



    meshdata.Close()

    with open("{}/SurfaceTemps.pkl".format(SimDict['CALC_DIR']),'wb') as f:
        pickle.dump(Data,f)



def ERMES_PV(VL,SimDict):
    Parameters = SimDict["Parameters"]
    ResFile = '{}/ERMES.rmed'.format(SimDict['PREASTER'])

    P = ERMESPP.Power(ResFile)
    SimDict['Data']['Power'] = P

    V = ERMESPP.Variation(ResFile)
    SimDict['Data']['Variation'] = V
