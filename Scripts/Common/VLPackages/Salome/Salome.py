#!/usr/bin/env python3

import os
import uuid
import pickle

from Scripts.Common.VLContainer import Container_Utils as Utils
import ContainerConfig


Dir = os.path.dirname(os.path.abspath(__file__))

def Run(Script, AddPath = [], DataDict = {}, OutFile=None, GUI=False, tempdir = '/tmp'):
    '''
    AddPath: Additional paths that Salome will be able to import from
    DataDict: a dictionary of the arguments that Salome will get
    OutFile: The log file you want to write stdout to
    GUI: Opens a new instance with GUI (useful for testing)
    tempdir: Location where pickled object can be written to
    '''

    # GetContainerInfo
    SalomeContainer = getattr(ContainerConfig,'Salome')


    # Add paths provided to python path for subprocess
    AddPath = [AddPath] if type(AddPath) == str else AddPath
    PyPath = AddPath + ['/home/ibsim/VirtualLab',Dir]
    PyPath = ":".join(PyPath)

    _argstr = []
    if DataDict:
        pth = "{}/DataDict_{}.pkl".format(tempdir,uuid.uuid4())
        with open(pth,'wb') as f:
            pickle.dump(DataDict,f)
        _argstr.append('DataDict={}'.format(pth))
    argstr = ",".join(_argstr)

    GUIflag = 'g' if GUI else 't'
   
    Wrapscript = "{}/SalomeExec.sh".format(Dir)
    command = "{} -c {} -f {} -a {} -p {} -r {} ".format(Wrapscript, SalomeContainer.Command, Script, argstr, PyPath, GUIflag)
                                                         
    RC = Utils.Exec_Container(SalomeContainer.ContainerFile,command,SalomeContainer.bind)
    return RC
