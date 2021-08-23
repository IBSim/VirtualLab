from subprocess import Popen
import sys
import VLconfig

Exec = getattr(VLconfig,'ERMESExec','ERMESv12.5')

def Run(Name,cwd=None):

    if True:
        ERMES_run = Popen("{} {}".format(Exec,Name),stdout=sys.stdout,stderr=sys.stderr,cwd=cwd,shell='TRUE')
    else:
        ERMES_run = Popen([Exec, Name], stdout=sys.stdout, stderr=sys.stderr, cwd=cwd)
    err = ERMES_run.wait()
    return err
