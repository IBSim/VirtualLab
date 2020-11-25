#!/usr/bin/env python3

import sys
sys.dont_write_bytecode=True
import os
from subprocess import Popen, PIPE, STDOUT
import uuid

class CodeAster():
    def __init__(self,super,**kwargs):
        self.Exec = super.ASTER_DIR # How to call CodeAster (can be changed for different versions etc.)

        self.TMP_DIR = super.TMP_DIR

        self.Logger = super.Logger
        self.Exit = super.Exit
        self.LogFile = super.LogFile

        self.mode = super.mode
        # AddPath will always add these paths to salome environment
        self.AddPath = kwargs.get('AddPath',[]) + ["{}/VLPackages/CodeAster".format(super.COM_SCRIPTS)]

    def ExportWriter(self,ExportFile,CommFile,MeshFile,ResultsDir,MessFile,**kwargs):
        # Create export file and write to file

        # CodeAster settings
        mpi_cpu = kwargs.get('mpi_nbcpu',1)
        mpi_nd = kwargs.get('mpi_nbnoeud',1)
        ncpus = kwargs.get('ncpus',1)
        memory = kwargs.get('memory',2)
        time = kwargs.get('time',99999)
        version = kwargs.get('version','stable')
        mode = kwargs.get('mode','batch')
        actions = kwargs.get('actions','make_etude')

        Settings='P actions {}\n'\
        'P mode {}\n'\
        'P version {}\n'\
        'P time_limit {}\n'\
        'P mpi_nbcpu {}\n'\
        'P mpi_nbnoeud {}\n'\
        'P ncpus {}\n'\
        'P memory_limit {!s}\n'\
        .format(actions,mode,version,time,mpi_cpu,mpi_nd,ncpus,float(1024*memory))

        Paths = 'F mmed {} D  20\n'\
        'F comm {} D  1\n'\
        'F mess {} R  6\n'\
        'R repe {} R  0\n'\
        .format(MeshFile,CommFile,MessFile,ResultsDir)

        with open(ExportFile,'w+') as e:
        	e.write(Settings+Paths)


    def Run(self, ExportFile, StudyDict, **kwargs):
        AddPath = kwargs.get('AddPath',[])

        AddPath = [AddPath] if type(AddPath) == str else AddPath
        PythonPath = ["{}:".format(path) for path in AddPath+self.AddPath]
        PythonPath = ["PYTHONPATH="] + PythonPath + ["$PYTHONPATH;export PYTHONPATH;export PYTHONDONTWRITEBYTECODE=1;"]
        PythonPath = "".join(PythonPath)

        OutFile = kwargs.get('OutFile', "")
        if 'OutFile' in kwargs:
            Output = " >{} 2>&1".format(kwargs['OutFile'])
        else : Output = " "

        if self.mode == 'Interactive':
            errfile = '{}/Aster.txt'.format(StudyDict['TMP_CALC_DIR'])
            command = "xterm -hold -T 'Study: {}' -sb -si -sl 2000 "\
            "-e '{} {}; echo $? >'{}".format(kwargs.get('Name',ExportFile),self.Exec, ExportFile, errfile)
        else:
            command = "{} {} {}".format(self.Exec, ExportFile, Output)

        # Start Aster subprocess
        proc = Popen(PythonPath + command , shell='TRUE')
        return proc
