import sys
sys.dont_write_bytecode=True
import numpy as np
import os
import traceback
import copy
from types import SimpleNamespace as Namespace
import pickle
from importlib import reload
from contextlib import redirect_stderr, redirect_stdout
import pathos.multiprocessing as pathosmp
from pyina.launchers import MpiPool

def PoolWrap(fn,VL,Dict,*args):
    # This function assumes that the first argument of args is the VL instance and
    # the second is a dictionary of relevant information created in Setup
    # Try and get name & log file if standard convention has been followed
    # the relevant dictionary will be the second argument
    if type(Dict)==dict:
        Name = Dict.get('Name',None)
        LogFile = Dict.get('LogFile',None)
    else : Name, LogFile = None, None

    Returner = Namespace()
    OrigDict = copy.deepcopy(Dict)
    try:
        if LogFile:
            # Output is piped to LogFile
            print("Running {}.\nOutput is piped to {}.\n".format(Name, LogFile))
            LogDir = os.path.dirname(LogFile)
            os.makedirs(LogDir,exist_ok=True)
            with open(LogFile,'w') as f:
                with redirect_stdout(f), redirect_stderr(f):
                    err = fn(VL,Dict,*args)
        else:
            print("Running {}.\n".format(Name))
            err = fn(VL,Dict,*args)

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
    except (Exception,SystemExit) as e:
        exc = e
        trb = traceback.format_exception(etype=type(exc), value=exc, tb = exc.__traceback__)
        err = "".join(trb)

        mess = "{} finishes with errors.".format(Name)
        return exc
    finally:
        if LogFile:
            # if error write it to log file
            if err:
                mess += " See the output file for more details.".format(LogFile)
                with open(LogFile,'a') as f:
                    f.write(str(err))
        elif err:
            mess += "{}\n".format(err)

        print(mess,flush=True)

def PoolReturn(Dicts,Returners):
    cpDicts = copy.deepcopy(Dicts)
    PlError = []
    for i, (Dict,Returner) in enumerate(zip(cpDicts,Returners)):
        Name = Dict['Name']
        if isinstance(Returner,Exception) or isinstance(Returner,SystemExit):
            PlError.append(Name)
            continue
        if Returner.Error:
            PlError.append(Name)
        if hasattr(Returner,'Dict'):
            Dicts[i].update(Returner.Dict)

    return PlError

def VLPool(VL,fnc,Dicts,Args=[],launcher='Sequential',N=1,onall=True):
    fnclist = [fnc]*len(Dicts)
    PoolArgs = [[VL]*len(Dicts),Dicts] + Args
    if launcher == 'Sequential':
        # Run studies one after the other
        Res = []
        for args in zip(*PoolArgs):
            ret = PoolWrap(fnc,*args)
            Res.append(ret)
    elif launcher == 'Process':
        # Run studies in parallel of N using pathos. Only works on single nodes.
        # Reloading pathos ensures any new paths added to sys.path are included
        pmp = reload(pathosmp)
        pool = pmp.ProcessPool(nodes=N, workdir=VL.TEMP_DIR)
        Res = pool.map(PoolWrap,fnclist, *PoolArgs)
    elif launcher == 'MPI':
        # Run studies in parallel of N using pyina. Works for multi-node clusters.
        # If available add 1 to work as a 'delegator'
        if not onall and NumThreads > N: N=N+1
        # Ensure that sys.path is the same for pyinas MPI subprocess
        PyPath_orig = os.environ.get('PYTHONPATH',"")
        addpath = set(sys.path) - set(VL._pypath) # group subtraction
        addpath = ":".join(addpath) # write in unix style
        # Update PYTHONPATH is os
        os.environ["PYTHONPATH"] = "{}:{}".format(addpath,PyPath_orig)

        # Run functions in parallel of N using pyina
        pool = MpiPool(nodes=N,source=True, workdir=VL.TEMP_DIR)
        Res = pool.map(PoolWrap,fnclist, *PoolArgs, onall=onall)

        # Reset environment back to original
        os.environ["PYTHONPATH"] = PyPath_orig

    # Check if errors have been returned & update dictionaries
    Errorfnc = PoolReturn(Dicts,Res)
    return Errorfnc
