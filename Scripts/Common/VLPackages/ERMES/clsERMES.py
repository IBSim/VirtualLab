from subprocess import Popen
import sys

class ERMES():
    def __init__(self):
        self.Exec = 'ERMESv12.5'

    def Run(self,Name,**kwargs):
        cwd = kwargs.get('cwd',None)
        # ERMES_run = Popen(Ermesstr, stdout=sys.stdout, stderr=sys.stdout, shell = 'TRUE')
        ERMES_run = Popen([self.Exec, Name], stdout=sys.stdout, stderr=sys.stderr, cwd=cwd)
        err = ERMES_run.wait()
        return err
