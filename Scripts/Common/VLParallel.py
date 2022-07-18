import sys
import os
import traceback
import copy
from types import SimpleNamespace as Namespace
import pickle
import inspect
from contextlib import redirect_stderr, redirect_stdout

import numpy as np

from Scripts.Common import Analytics
from Scripts.Common.tools import Paralleliser
import VLconfig

def PoolWrap(fn,VL,Dict,*args):
    # Try and get name & log file if standard convention has been followed
    if type(Dict)==dict:
        Name = Dict.get('Name',None)
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
            with open(LogFile,'w') as f:
                with redirect_stdout(f), redirect_stderr(f):
                    err = fn(VL,Dict,*args)
        else:
            print("Running {}.\n".format(Name),flush=True)
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
        Name = Dict['Name']
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

def VLPool(VL,fnc,Dicts,Args=[],launcher=None,N=None):

    fnclist = [fnc]*len(Dicts)
    PoolArgs = [[VL]*len(Dicts),Dicts] + Args

    if not N: N = VL._NbJobs
    if not launcher: launcher = VL._Launcher

    try:
        if launcher == 'Sequential' or len(Dicts)==1:
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
            Res=list(Res)
            pool.terminate()
        elif launcher in ('MPI','MPI_Worker'):
            # Run studies in parallel of N using pyina. Works for multi-node clusters.
            # onall specifies if there is a worker. True = no worker
            if launcher == 'MPI' or N==1: onall = True # Cant have worker if N is 1
            else: onall = False

            # Ensure that sys.path is the same for pyinas MPI subprocess
            PyPath_orig = os.environ.get('PYTHONPATH',"")
            addpath = set(sys.path) - set(VL._pypath) # group subtraction
            addpath = ":".join(addpath) # write in unix style
            # Update PYTHONPATH is os
            os.environ["PYTHONPATH"] = "{}:{}".format(addpath,PyPath_orig)

        Res = Paralleliser(PoolWrap,PoolArgs, method=launcher, nb_parallel=N, **kwargs)
    except KeyboardInterrupt as e:
        VL.Cleanup()

        exc = e
        trb = traceback.format_exception(etype=type(exc), value=exc, tb = exc.__traceback__)

        sys.exit("".join(trb))

	# Function to analyse usage of VirtualLab to evidence impact for
	# use in future research grant applications. Can be turned off via
	# VLconfig.py. See Scripts/Common/Analytics.py for more details.
    if VLconfig.VL_ANALYTICS=="True":
        frame = inspect.stack()[1]
        module = inspect.getmodule(frame[0])
        vltype = os.path.splitext(os.path.basename(module.__file__))[0]

        Category = "{}_{}".format(VL.Simulation,vltype)
        Action = "NJob={}_NCore={}_NNode=1".format(len(Dicts),N)

        # ======================================================================
        # Send information abotu current job
        Analytics.Run(Category,Action,VL._ID)

        # ======================================================================
        # Update Analytics dictionary with new information
        # Create dictionary, if one isn't defined already
        if not hasattr(VL,'_Analytics'): VL._Analytics = {}
        # Add information to dictionary
        if vltype not in VL._Analytics:
            VL._Analytics[vltype] = len(Dicts)
        else:
            VL._Analytics[vltype] += len(Dicts)


    # Check if errors have been returned & update dictionaries
    Errorfnc = PoolReturn(Dicts,Res)
    return Errorfnc
