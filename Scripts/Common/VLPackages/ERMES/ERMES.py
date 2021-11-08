from subprocess import Popen
import sys
import os
import VLconfig

Exec = getattr(VLconfig,'ERMESExec','ERMESv12.5')

Container = getattr(VLconfig,'ERMESContainer',None)
if Container:
    import ContainerConfig
    ERMESContainer = getattr(ContainerConfig,Container)
def Run(Name, Append=False):
    dirname = os.path.dirname(Name)
    LogFile = "{}/ERMESLog".format(dirname)
    if Append:
        tee = "| tee -a {}".format(LogFile)
    else:
        tee = "| tee {}".format(LogFile)

    if Container:
        command = "{} {} {} {}".format(ERMESContainer.Call,ERMESContainer.ERMESExec,Name,tee)
    else:
        command = "{} {} {}".format(Exec,Name,tee)

    ERMES_run = Popen(command,stdout=sys.stdout,stderr=sys.stderr,cwd=dirname,shell='TRUE')
    err = ERMES_run.wait()
    return err
