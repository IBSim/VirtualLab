#!/usr/bin/env python3
import os

from Scripts.VLPackages.Salome.API import Run as _Run

Dir = os.path.dirname(os.path.abspath(__file__))

def Run(Script, ContainerInfo=None, AddPath = [], DataDict = {}, GUI=False, tempdir = '/tmp'):
    AddPath.append(Dir)
    _Run(Script, ContainerInfo=ContainerInfo, AddPath = AddPath, DataDict = DataDict, GUI=GUI, tempdir = tempdir)

def RunEval(Script, EvalList, ContainerInfo=None, AddPath = [], DataDict = {}, GUI=False, tempdir = '/tmp'):

    if EvalList: # add this to DataDict with a specific key which is picked up by another function
        DataDict['_PV_arg'] = EvalList
    
    Run(Script, ContainerInfo=ContainerInfo, AddPath = AddPath, DataDict = DataDict, GUI=GUI, tempdir = tempdir)

