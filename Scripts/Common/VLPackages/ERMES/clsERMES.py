from subprocess import Popen
import sys
import VLconfig

class ERMES():
    def __init__(self):
        self.Exec = getattr(VLconfig,'ERMESExec','ERMESv12.5')

    def Run(self,Name,**kwargs):
        cwd = kwargs.get('cwd',None)
        if True:
            ERMES_run = Popen("{} {}".format(self.Exec,Name),stdout=sys.stdout,stderr=sys.stderr,cwd=cwd,shell='TRUE')
        else:
            ERMES_run = Popen([self.Exec, Name], stdout=sys.stdout, stderr=sys.stderr, cwd=cwd)
        err = ERMES_run.wait()
        return err
