#!/usr/bin/env python3
import os

from Scripts.VLPackages.Salome import API as Salome 

Dir = os.path.dirname(os.path.abspath(__file__))

def Run(Script, ContainerInfo=None, AddPath = [], DataDict = {}, GUI=False, tempdir = '/tmp'):
    AddPath.append(Dir)
    Salome.Run(Script, ContainerInfo=ContainerInfo, AddPath = AddPath, DataDict = DataDict, GUI=GUI, tempdir = tempdir)

def RunEval(Script, EvalList, ContainerInfo=None, AddPath = [], DataDict = {}, GUI=False, tempdir = '/tmp'):

    if EvalList: # add this to DataDict with a specific key which is picked up by another function
        DataDict['_PV_arg'] = EvalList
    
    Run(Script, ContainerInfo=ContainerInfo, AddPath = AddPath, DataDict = DataDict, GUI=GUI, tempdir = tempdir)

def ShowMED(MedDict,**kwargs):
    # remove DataDict if its in kwargs as it will be expanded later on
    DataDict = kwargs.pop('DataDict') if 'DataDict' in kwargs else {}
    DataDict['_ShowMED_'] = MedDict

    Script = "{}/ShowMED.py".format(Dir)

    Run(Script,DataDict=DataDict,**kwargs)


def OpenGUI():
    Script = "{}/OpenGUI.py".format(Dir)
    Run(Script,GUI=True)
    