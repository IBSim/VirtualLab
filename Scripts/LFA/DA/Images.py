
import os
import numpy as np
import h5py
from natsort import natsorted

def Single(VL,DADict):
    Parameters = DADict['Parameters']
    ResultNames = getattr(Parameters,'ResultNames',[])
    ResDir = "{}/{}".format(VL.PROJECT_DIR, DADict["Name"])

    GlobalRange = [np.inf, -np.inf]
    ResData = {}
    for ResName in natsorted(os.listdir(ResDir)):
        ResSubDir = "{}/{}".format(ResDir,ResName)
        if not os.path.isdir(ResSubDir): continue
        if ResultNames and ResName not in ResultNames: continue

        ResFile = '{}/Aster/Thermal.rmed'.format(ResSubDir)
        g = h5py.File(ResFile, 'r')
        gRes = g['/CHA/Temperature']
        Steps = list(gRes.keys())
        Time = np.array([gRes[step].attrs['PDT'] for step in Steps])
        CT_Ix = np.argmin(np.abs(Time-Parameters.CaptureTime))

        resTemp = gRes['{}/NOE/MED_NO_PROFILE_INTERNAL/CO'.format(Steps[CT_Ix])][:]
        # Update GlobalRange with global min and max values
        GlobalRange = [min(min(resTemp),GlobalRange[0]),max(max(resTemp),GlobalRange[1])]

        g.close()

        ResData[ResName] = {'File':ResFile,
                            'Time':Time[CT_Ix],
                            'ImageDir':"{}/PostAster".format(ResSubDir)}

    DADict['ResData'] = ResData
    DADict['GlobalRange'] = GlobalRange

    print('Creating images using ParaViS')

    GUI = getattr(Parameters, 'PVGUI', False)
    ParaVisFile = "{}/ParaVis.py".format(os.path.dirname(os.path.abspath(__file__)))
    RC = VL.Salome.Run(ParaVisFile, DataDict=DADict, GUI=GUI)
    if RC:
        return "Error in Salome run"
