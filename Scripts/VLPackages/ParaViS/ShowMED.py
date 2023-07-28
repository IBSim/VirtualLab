import sys
import os
sys.dont_write_bytecode=True

from SalomeFunc import GetArgs

DataDict = GetArgs()

if '_ShowMED_' not in DataDict:
    raise Exception('OpenMED.py will only work if its called using the OpenMED function in the API')

# import these after the check as they are slow to load
import SalomePyQt
import pvsimple
import PVFunc

MedDict = DataDict['_ShowMED_']

renderView1 = pvsimple.GetActiveViewOrCreate('RenderView')

ResDict = {}
for name, path in MedDict.items():
    Res = PVFunc.OpenMED(path)
    pvsimple.RenameSource(name, Res)
    ResDict[name]=Res

Display = pvsimple.Show(Res, renderView1)
renderView1.ResetCamera()

SalomePyQt.SalomePyQt().activateModule('ParaViS')
