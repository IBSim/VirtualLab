import os
from importlib import reload

import pathos.multiprocessing as pathosmp
from pyina.launchers import MpiPool

def Paralleliser(fnc, args, method='sequential', nb_parallel=1, **kwargs):
    '''
    Evaluate function 'fnc' for a range of agruments using a chosen method.
    Methods available are:
    - sequential (no parallelisation)
    - pathos (single node parallelisation)
    - pyina (multi node parallelisation using mpi).

    nb_parallel: number of jobs to run in parallel (pathos & pyina only)
    workdir: directory where temporary files are created (pathos & pyina only)
    addpath: additional paths needed for python (pyina only)
    '''

    NbEval = len(args[0]) # Number of function evaluations required

    if method.lower() == 'sequential' or NbEval==1:
        Res = []
        for arg in zip(*args):
            ret = fnc(*arg)
            Res.append(ret)
    elif method.lower() == 'process':
        pmp = reload(pathosmp)
        workdir = kwargs.get('workdir',None)
        pool = pmp.ProcessPool(nodes=nb_parallel, workdir=workdir)
        Res = pool.map(fnc, *args)
        Res = list(Res)
        pool.terminate()
    elif method.lower() in ('mpi','mpi_worker'):
        # mpi_worker keeps one worker free to assign jobs.
        if method.lower() == 'mpi' or nb_parallel==1: # Cant have worker if N is 1
            onall = True
        else: onall = False

        workdir = kwargs.get('workdir',None)
        addpath = kwargs.get('addpath',[])
        source = kwargs.get('source',True)

        # Ensure that sys.path is the same for pyinas MPI subprocess
        PyPath_orig = os.environ.get('PYTHONPATH',"")

        if addpath:
            # Update PYTHONPATH with addpath for matching environment
            os.environ["PYTHONPATH"] = "{}:{}".format(":".join(addpath), PyPath_orig)

        # Run functions in parallel of N using pyina
        pool = MpiPool(nodes=nb_parallel, source=source, workdir=workdir)
        Res = pool.map(fnc, *args, onall=onall)
        Res = list(Res)

        # Reset environment back to original
        os.environ["PYTHONPATH"] = PyPath_orig

    return Res
