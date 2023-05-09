import sys
import os
import traceback
import copy
from types import SimpleNamespace as Namespace, MethodType
import pickle
from contextlib import redirect_stderr, redirect_stdout
import numpy as np
import time

from subprocess import Popen, call
import tempfile
import logging
import dill

from pyina.launchers import MpiPool
from pyina.tools import which_python, which_mpirun, which_strategy

from Scripts.Common.VLContainer import Container_Utils as Utils
from Scripts.Common.tools.Paralleliser import Paralleliser, _fn_wrap_kwargs


def _time_fn(fn,*args,**kwargs):
    st = time.time()
    err = fn(*args,**kwargs)
    end = time.time()
    walltime = time.strftime("%H:%M:%S",time.gmtime(end-st))
    print("\n################################\n\n"\
          "Wall time for analysis: {}\n\n"\
          "################################".format(walltime))
    return err

def PoolWrap(fn,VL,Dict,*args,**kwargs):
    # Try and get name & log file if standard convention has been followed
    if type(Dict)==dict:
        Name = Dict.get('_Name',None)
        LogFile = Dict.get('LogFile',None)
    else : Name, LogFile = None, None

    Returner = Namespace()
    OrigDict = copy.deepcopy(Dict)
    err=None
    try:
        if LogFile:
            # Output is piped to LogFile
            print("Running {}.\nOutput is piped to {}.\n".format(Name, LogFile),flush=True)
            LogDir = os.path.dirname(LogFile)
            os.makedirs(LogDir,exist_ok=True)
            with open(LogFile,'w') as f: pass # make blank file
            with open(LogFile,'a') as f:
                with redirect_stdout(f), redirect_stderr(f):
                    err = _time_fn(fn,VL,Dict,*args,**kwargs)
        else:
            print("Running {}.\n".format(Name),flush=True)
            err = _time_fn(fn,VL,Dict,*args,**kwargs)

        if not err: mess = "{} completed successfully.\n".format(Name)
        else: mess = "{} finishes with errors.\n".format(Name)

        Returner.Error = err # will be None if everything has run smoothly
        if not OrigDict == Dict:
            # Attach dictionary to Returner to update VL class
            Returner.Dict = Dict
            # Save information in Data to location specified by __file__
            Data = Dict.get('Data',{})
            if '__file__' in Data and len(Data)>1:
                with open(Data['__file__'],'wb') as f:
                    pickle.dump(Data,f)

        return Returner
    # except (Exception,SystemExit,KeyboardInterrupt) as e:
    except (Exception,SystemExit) as e:
        exc = e
        trb = traceback.format_exception(etype=type(exc), value=exc, tb = exc.__traceback__)
        err = "".join(trb)
        mess = "{} finishes with errors.".format(Name)
        return exc
    finally:
        if err and LogFile:
            mess += " See the output file for more details.".format(LogFile)
            with open(LogFile,'a') as f:
                f.write(str(err))
        elif err:
            mess += "{}\n".format(err)

        print(mess)

def PoolReturn(Dicts,Returners):
    '''
    Check what's been returned by 'PoolWrap' to see if it has been successful.
    '''
    cpDicts = copy.deepcopy(Dicts)
    PlError = []
    for i, (Dict,Returner) in enumerate(zip(cpDicts,Returners)):
        Name = Dict['_Name']
        if isinstance(Returner,Exception) or isinstance(Returner,SystemExit):
            # Exception thrown
            PlError.append(Name)
            continue
        if Returner.Error:
            # Error message returned by the function 'fnc' used by PoolWrap.
            PlError.append(Name)
        if hasattr(Returner,'Dict'):
            # If a dictionary is attached to Returner it means new
            # information has been added, so we update the original dictionary.
            Dicts[i].update(Returner.Dict)

    return PlError


def containermap(self, func, *args, **kwds):
    log = logging.getLogger("mpi")
    log.addHandler(logging.StreamHandler())
    _SAVE = [False]

    # set strategy
    if self.scatter:
        kwds['onall'] = kwds.get('onall', True)
    else:
        kwds['onall'] = kwds.get('onall', True) #XXX: has pickling issues
    config = {}
    config['program'] = which_strategy(self.scatter, lazy=True)

    # serialize function and arguments to files
    modfile = self._modularize(func)
    argfile = self._pickleargs(args, kwds)
    # Keep the above handles as long as you want the tempfiles to exist
    if _SAVE[0]:
        _HOLD.append(modfile)
        _HOLD.append(argfile)
    # create an empty results file
    resfilename = tempfile.mktemp(dir=self.workdir)
    # process the module name
    modname = self._modulenamemangle(modfile.name)
    # build the launcher's argument string
    config['progargs'] = ' '.join([modname, argfile.name, \
                                   resfilename, self.workdir])

    #XXX: better with or w/o scheduler baked into command ?
    #XXX: better... if self.scheduler: self.scheduler.submit(command) ?
    #XXX: better if self.__launch modifies command to include scheduler ?
    if _SAVE[0]:
        self._save_in(modfile.name, argfile.name) # func, pickled input
    # create any necessary job files
    if self.scheduler: config.update(self.scheduler._prepare())
    ######################################################################
    # build the launcher command

    command = self._launcher(config)

    log.info('(skipping): %s' % command)
    if log.level == logging.DEBUG:
        error = False
        res = []
    else:

        try:
            error = Utils.MPI_Container({'ContainerName':'Manager'}, command)    
            #subproc = Popen([command],shell=True)
            #error = subproc.wait()  # block until all done


            # just to be sure... here's a loop to wait for results file ##
            maxcount = self.timeout; counter = 0
            #print "before wait"
            while not os.path.exists(resfilename):
                call('sync', shell=True)
                from time import sleep
                sleep(1); counter += 1
                if counter >= maxcount:
                    print("Warning: exceeded timeout (%s s)" % maxcount)
                    break
            #print "after wait"
            # read result back
            res = dill.load(open(resfilename,'rb'))

            #print "got result"
        except:
            error = True
           
    ######################################################################

    # cleanup files
    if _SAVE[0] and log.level == logging.WARN:
        self._save_out(resfilename) # pickled output
    self._cleanup(resfilename, modfile.name, argfile.name)
    if self.scheduler and not _SAVE[0]: self.scheduler._cleanup()
    if error:
        raise IOError("launch failed: %s" % command)
    return res


def Container_MPI(fnc, VL, args_list, kwargs_list=[], nb_parallel=1, onall=True, **kwargs):
  
    NbEval = len(args_list)

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
    pool.map = MethodType(containermap, pool)


    if kwargs_list == []:
        Res = pool.map(fnc, *args_list,onall=onall)
    else:
        # pass kwargs as additional args which is picked up by _fn_wrap_kwargs
        fncs = [fnc]*NbEval 
        Res = pool.map(_fn_wrap_kwargs, fncs, *args_list, kwargs_list, onall=onall)

    Res = list(Res)

    # Reset environment back to original
    os.environ["PYTHONPATH"] = PyPath_orig
    
    return Res



def VLPool(VL,fnc,Dicts,args_list=[],kwargs_list=[],launcher=None,N=None):

    if args_list:
        assert len(args_list)==len(Dicts)
    if kwargs_list:
        assert len(kwargs_list)==len(Dicts)

    analysis_names = list(Dicts.keys())
    analysis_data = []
    for analysis_name in analysis_names:
        Dicts[analysis_name]['_Name'] = analysis_name
        analysis_data.append(Dicts[analysis_name])

    if not N: N = VL._NbJobs
    if not launcher: launcher = VL._Launcher

    VL_dict = VL.__dict__.copy()
    VL_dict = Namespace(**VL_dict)

    # create list fof arguments
    PoolArgs = []
    for i,_dict in enumerate(analysis_data):
        a = [fnc,VL_dict,_dict]
        # add args_list info, if it exists
        if args_list:
            _arg = args_list[i]
            if type(_arg) in (list,tuple): a.extend(_arg)
            else: a.append(_arg)
        PoolArgs.append(a)

    try:
        if launcher.lower()=='process':
            kwargs = {'workdir':VL.TEMP_DIR}
        elif launcher.lower() in ('mpi','mpi_worker'):
            kwargs = {'workdir':VL.TEMP_DIR,
                      'addpath':set(sys.path)-set(VL._pypath)}
            VL = Namespace()
        else: kwargs = {}

   
        if launcher.lower().startswith('mpi'):
            if launcher.lower() == 'mpi' or N==1: onall = True
            else: onall = False
            Res = Container_MPI(PoolWrap,VL,PoolArgs,kwargs_list=kwargs_list, nb_parallel=N, onall=onall, **kwargs)
        else:
            Res = Paralleliser(PoolWrap,VL,PoolArgs,kwargs_list=kwargs_list, method=launcher, nb_parallel=N,**kwargs)
        
        
    except KeyboardInterrupt as e:
        VL.Cleanup()

        exc = e
        trb = traceback.format_exception(etype=type(exc), value=exc, tb = exc.__traceback__)

        sys.exit("".join(trb))

    # Check if errors have been returned & update dictionaries
    Errorfnc = PoolReturn(analysis_data,Res)

    return Errorfnc