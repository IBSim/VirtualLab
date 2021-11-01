from subprocess import Popen
import sys
import os
import VLconfig

Exec = getattr(VLconfig,'ERMESExec','ERMESv12.5')

def Run(Name, Append=False):

    if True:
        dirname = os.path.dirname(Name)
        LogFile = "{}/ERMESLog".format(dirname)
        if Append:
            tee = "| tee -a {}".format(LogFile)
        else:
            tee = "| tee {}".format(LogFile)
        ERMES_run = Popen("{} {} {}".format(Exec,Name,tee),stdout=sys.stdout,stderr=sys.stderr,cwd=dirname,shell='TRUE')
    else:
        ERMES_run = Popen([Exec, Name], stdout=sys.stdout, stderr=sys.stderr, cwd=cwd)
    err = ERMES_run.wait()
    return err
