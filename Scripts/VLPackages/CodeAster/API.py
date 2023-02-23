#!/usr/bin/env python3

import sys
sys.dont_write_bytecode=True
import os
from subprocess import Popen, PIPE, STDOUT
import uuid
import shutil

from Scripts.Common.VLContainer import Container_Utils as Utils
from Scripts.VLPackages.ContainerInfo import GetInfo

CADir = os.path.dirname(os.path.abspath(__file__))

''' 
This is an API for the VL_Manager container to send information to the server
to run analysis using the CodeAster package (which is installed in a different container). 
'''

def ExportWriter(ExportFile,CommFile,MeshFile,ResultsDir,MessFile,Settings):
    # Create export file and write to file

    # CodeAster settings
    mpi_cpu = Settings.get('mpi_nbcpu',1)
    mpi_nd = Settings.get('mpi_nbnoeud',1)
    ncpus = Settings.get('ncpus',1)
    memory = Settings.get('memory',2) # Giggabytes
    time = Settings.get('time',99999) # seconds
    version = Settings.get('version','stable')
    mode = Settings.get('mode','batch')
    actions = Settings.get('actions','make_etude')
    rep_trav = Settings.get('rep_trav',None)
    # mpi_cpu=1


    Settings='P actions {}\n'\
    'P mode {}\n'\
    'P version {}\n'\
    'P time_limit {}\n'\
    'P mpi_nbcpu {}\n'\
    'P mpi_nbnoeud {}\n'\
    'P ncpus {}\n'\
    'P memory_limit {!s}\n'\
    .format(actions,mode,version,time,mpi_cpu,mpi_nd,ncpus,float(1024*memory))
    if rep_trav:
        Settings+="P rep_trav {}\n".format(rep_trav)

    Paths = 'F mmed {} D  20\n'\
    'F comm {} D  1\n'\
    'F mess {} R  6\n'\
    'R repe {} R  0\n'\
    .format(MeshFile,CommFile,MessFile,ResultsDir)

    with open(ExportFile,'w+') as e:
    	e.write(Settings+Paths)

#def RunXterm(ExportFile, AddPath = [], OutFile=None, tempdir = '/tmp'):

#    AddPath = [AddPath] if type(AddPath) == str else AddPath
#    PyPath = ["{}:".format(path) for path in AddPath+[CADir]]
#    PyPath = "".join(PyPath)

#    # GetContainerInfo
#    CAContainer = getattr(ContainerConfig,'CodeAster')

#    WrapScript = "{}/AsterExec.sh".format(CADir)
#    
#    errfile = "{}/{}".format(tempdir, uuid.uuid4())
#    xterm_command = "xterm -hold -T 'Study: {0}' -sb -si -sl 2000 "\
#    "-e '{1} {0}; echo $? >{2}';exit $(cat {2})".format(ExportFile, CAContainer.Command, errfile)
#    command = '''{} -c "{}" -p {} '''.format(WrapScript, xterm_command, PyPath)
#    print(command)
#    command=ExportFile
#    RC = Utils.Exec_Container(CAContainer.ContainerFile,command,CAContainer.bind)
#    return RC

def RunXterm(ExportFile, AddPath = [], OutFile=None, tempdir = '/tmp'):
    Run(ExportFile,AddPath=AddPath)

def Run(ExportFile, ContainerInfo=None, AddPath = []):

    if ContainerInfo is None:
        # Get default container info
        ContainerInfo = GetInfo('CodeAster')
        
    AddPath = [AddPath] if type(AddPath) == str else AddPath
    PyPath = ["{}:".format(path) for path in AddPath+[CADir]]
    PyPath = "".join(PyPath)

    WrapScript = "{}/AsterExec.sh".format(CADir)
    command = "{} -c {} -f {} -p {} ".format(WrapScript,ContainerInfo['Command'], ExportFile, PyPath)
    
    RC = Utils.Exec_Container(ContainerInfo, command)

    return RC




def RunMPI(N, ExportFile, rep_trav, LogFile, ResDir, AddPath = [], OutFile=None):

    AddPath = [AddPath] if type(AddPath) == str else AddPath
    PyPath = ["{}:".format(path) for path in AddPath+[CADir]]
    PyPath = "".join(PyPath)
    env = {**os.environ, 'PYTHONPATH': PyPath + os.environ.get('PYTHONPATH','')}
    Output = ">>{} 2>&1".format(OutFile) if OutFile else ""

    # ==========================================================================
    # Create CodeAster environment in 'rep_trav' directory
    if Container:
        command = "{} {} {} ".format(AsterContainer.Call,AsterContainer.AsterExec, ExportFile)
    else:
        command = "{} {} ".format(Exec,ExportFile)

    proc1 = Popen(command, shell='TRUE', stdout=sys.stdout, stderr=sys.stderr, env=env)
    err = proc1.wait()
    if err: return err

    # ==========================================================================
    # Create file which loads paths
    # Copy this file as this is what CodeAster calls
    shutil.copy("{}/global/fort.1.1".format(rep_trav),"{}/global/fort.1".format(rep_trav))

    # Create script to launch. mpi_script.sh is created by CodeAster but certain
    # paths need to be loaded before hand
    if Container:
        AsterPath = getattr(AsterContainer, 'Path',
                            os.path.dirname(os.path.dirname(AsterContainer.AsterExec)))
    else:
        AsterPath = os.path.dirname(os.path.dirname(Exec))

    LaunchScript = "{}/Launch.sh".format(rep_trav)
    LaunchString = ". {0}/etc/codeaster/profile.sh\n"\
                   ". {0}/14.4_mpi/share/aster/profile.sh\n"\
                   ". {1}/global/profile_tmp.sh\n\n"\
                   "{1}/global/mpi_script.sh".format(AsterPath,rep_trav)

    with open(LaunchScript,'w') as f:
        f.write(LaunchString)
    os.chmod(LaunchScript,0o777)

    # =========================================================================
    # Launch CodeAster with MPI
    if Container:
        command = "mpirun -np {} {} {} | tee {}/AsterLog".format(N,AsterContainer.Call,
                                                                 LaunchScript,rep_trav)
    else:
        command = "mpirun -np {} {} | tee {}/AsterLog".format(N,LaunchScript,rep_trav)

    proc2 = Popen(command, shell='TRUE', stdout=sys.stdout, stderr=sys.stderr, env=env)
    err = proc2.wait()

    # Copy AsterLog to LogFile location
    shutil.copy2("{}/AsterLog".format(rep_trav),LogFile)

    # Copy context of REPE_OUT to results directory
    REPE_OUT = "{}/global/REPE_OUT".format(rep_trav)
    for dir_,_,files in os.walk(REPE_OUT):
        for file in files:
            src = os.path.join(dir_, file)
            relpath = os.path.relpath(dir_,REPE_OUT)
            dst = os.path.join(ResDir,relpath, file)
            shutil.copy(src,dst)


    return err
