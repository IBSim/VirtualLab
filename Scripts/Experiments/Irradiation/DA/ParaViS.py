#!/bin/bash

import sys
sys.dont_write_bytecode=True
import salome
import numpy as np

from importlib import import_module
import SalomeFunc
import os

'''
import SalomePyQt
SalomePyQt.SalomePyQt().activateModule('ParaViS')
'''
import pvsimple

pvsimple.ShowParaviewView()


DADict = SalomeFunc.GetArgs()
ResData = DADict['ResData']

CBmin,CBmax = DADict['GlobalRange']



conditions = ["TMax_tn", "VMMax_tn", "TMin_tn"]


def script(condition): 
    
    renderView1 = pvsimple.GetActiveViewOrCreate('RenderView')

    for Name,Data in ResData.items():
   
        thermalrmed= pvsimple.MEDReader(FileName=Data['File'])
        yieldrmed= pvsimple.MEDReader(FileName=Data['File1'])
        pvsimple.RenameSource(Name,thermalrmed)
        pvsimple.RenameSource(Name,yieldrmed)
        # show data in view
        pvsimple.SaveData("{}/vmis.vtm".format(Data['resDir']), thermalrmed)
        pvsimple.SaveData("{}/yield.vtm".format(Data['resDir']), yieldrmed)


  
    return None
for condition in conditions: 
    script(condition)
    sys.exit(0)   
