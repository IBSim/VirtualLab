import os
from importlib import reload
##############################################
# block to check for pathos/mpi
pathos_installed = True
mpi_installed = True 
try: 
    import pathos.multiprocessing as pathosmp
except ImportError:
    pathos_installed = False
try:  
    from pyina.launchers import MpiPool
except ImportError: 
    mpi_installed = False
##############################################

def _fn_wrap_kwargs(fn,*args):
    ''' wrapper function which enables kwargs to be passed to pyina and pathos'''
    kwargs = args[-1]
    args = args[:-1]
    return fn(*args,**kwargs)


def Paralleliser(VL,fnc, args_list, kwargs_list=None, method='sequential', nb_parallel=1, **kwargs):
    '''
    Evaluate function 'fnc' for a range of arguments using a chosen method.
    Methods available are:
    - sequential (no parallelisation)
    - pathos (single node parallelisation)
    - pyina (multi node parallelisation using mpi).

    nb_parallel: number of jobs to run in parallel (pathos & pyina only)
    workdir: directory where temporary files are created (pathos & pyina only)
    addpath: additional paths needed for python (pyina only)
    '''

    NbEval = len(args_list) # Number of function evaluations required
    # checks
    ############################################################
    # checks to see if mpi/pathos is requested but not installed
    if method.lower() in ('mpi','mpi_worker') and not mpi_installed:
        VL.Logger("********************************************\n",
              "WARNING: mpi is not installed in container\n",
              " Thus mpi can not be used. Runs will be\n",
              " performed sequentially.\n.",
              "********************************************", print=True)
        method = 'sequential'
    elif method.lower() == 'process' and not pathos_installed:
        VL.Logger("********************************************\n",
              "WARNING: pathos is not installed in container\n",
              " Thus process cannot be used. Runs will be \n",
              "performed sequentially.\n.",
              "********************************************",print=True)
        
        method = 'sequential'
    ###########################################################
    if method.lower() == 'sequential' or NbEval==1:
        Res = []
        for i, arg in enumerate(args_list):
            if kwargs_list is None:
                ret = fnc(*arg)
            else:
                ret = fnc(*arg,**kwargs_list[i])
            Res.append(ret)
    elif method.lower() == 'process' and pathos_installed:
        pmp = reload(pathosmp)
        workdir = kwargs.get('workdir',None)
        pool = pmp.ProcessPool(nodes=nb_parallel, workdir=workdir)
        args_list = list(zip(*args_list)) # change format of args for this
        if kwargs_list is None:
            Res = pool.map(fnc, *args_list)
        else:
            # pass kwargs as additional args which is picked up by _fn_wrap_kwargs
            Res = pool.map(_fn_wrap_kwargs,[fnc]*NbEval, *args_list, kwargs_list)
        Res = list(Res)
        pool.terminate()
    elif method.lower() in ('mpi','mpi_worker') and mpi_installed:
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

        args_list = list(zip(*args_list)) # change format of args for this
        # Run functions in parallel of N using pyina
        pool = MpiPool(nodes=nb_parallel, source=source, workdir=workdir)


        if kwargs_list is None:
            Res = pool.map(fnc, *args_list,onall=onall)
        else:
            # pass kwargs as additional args which is picked up by _fn_wrap_kwargs
            Res = pool.map(_fn_wrap_kwargs, [fnc]*NbEval, *args_list, kwargs_list, onall=onall)

        Res = list(Res)

        # Reset environment back to original
        os.environ["PYTHONPATH"] = PyPath_orig

    return Res



def _f(x):
    return x**2

def _f_kw(x,var=2,var2=6):
    return x**var + var2

def test():
    x = [[1],[2],[3],[4]]

    for method in ('sequential','process','mpi'):
        res = Paralleliser(_f,x,method=method)
        print(res)

    kw = [{'var':1,'var2':1},{'var':2},{'var':3},{'var':4}]
    for method in ('sequential','process','mpi'):
        res = Paralleliser(_f_kw,x,kw,method=method)
        print(res)

if __name__=='__main__':
    test()
