import sys
import os
sys.dont_write_bytecode=True
import salome
import numpy as np
from SalomeFunc import GetArgs
import SalomePyQt

kwargs = GetArgs()

# Connect to ParaVis server
from pvsimple import *

renderView1 = GetActiveViewOrCreate('RenderView')

ResDict = {}
for name, path in kwargs.items():
    if os.path.isfile(path):
        Res = MEDReader(FileName=path)
        RenameSource(name, Res)
        ResDict[name]=Res
Display = Show(Res, renderView1)
renderView1.ResetCamera()

SalomePyQt.SalomePyQt().activateModule('ParaViS')
