import os

from Scripts.Common.VLContainer import Container_Utils as Utils
from Scripts.Common.VLPackages.ContainerInfo import GetInfo

Dir = os.path.dirname(os.path.abspath(__file__))

def Run(AnalysisName, ContainerInfo=None, Append=False):
    if ContainerInfo is None:
        # Get default container info
        ContainerInfo = GetInfo('ERMES') 

    Wrapscript = "{}/ERMESExec.sh".format(Dir)
    command = "{} -c {} -f {} ".format(Wrapscript, ContainerInfo.Command, AnalysisName)

    RC = Utils.Exec_Container(ContainerInfo.ContainerFile, command, ContainerInfo.bind)
    return RC
