import pandas as pd
import os
import json
import numpy as np
import h5py
from Scripts.VLPackages.Salome.API import Run as SalomeRun
from Scripts.VLPackages.ParaViS import API as ParaVis
import sys
sys.dont_write_bytecode=True

import numpy as np
from importlib import import_module



def Single(VL,DADict):
  
    ResDir = "{}/{}/Aster".format(VL.PROJECT_DIR, DADict["_Name"])
    
    GlobalRange = [np.inf, -np.inf]
    ResData = {}
    for ResName in os.listdir(ResDir):
        ResSubDir = "{}/{}".format(ResDir,ResName)
      

        ResFile = '{}/vmis.rmed'.format(ResDir)
        ResFile1 = '{}/yieldstrength_cucrzr.rmed'.format(ResDir)
        ResData[ResName] = {'File':ResFile,
                            'File1':ResFile1,
                            'resDir':"{}".format(ResDir)
                           }
    DADict['ResData'] = ResData
    DADict['GlobalRange'] = GlobalRange
   
    print('Creating images using ParaViS')

  
    ParaVisFile = "{}/ParaViS.py".format(os.path.dirname(os.path.abspath(__file__)))
    RC = SalomeRun(ParaVisFile, DataDict=DADict, GUI=True)
    if RC:
        return "Error in Salome run"
