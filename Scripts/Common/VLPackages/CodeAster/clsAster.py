#!/usr/bin/env python3

import sys
sys.dont_write_bytecode=True
import os
from subprocess import Popen, PIPE, STDOUT

class CodeAster():
    def __init__(self,super,**kwargs):
        self.Exec = super.ASTER_DIR # How to call CodeAster (can be changed for different versions etc.)

        self.TMP_DIR = super.TMP_DIR

        self.Logger = super.Logger
        self.Exit = super.Exit
        self.LogFile = super.LogFile
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


    def Run(self,**kwargs):
        pass
