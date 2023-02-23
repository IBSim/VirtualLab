import os

from Scripts.Common.VLContainer import Container_Utils as Utils
from Scripts.VLPackages.ContainerInfo import GetInfo

Dir = os.path.dirname(os.path.abspath(__file__))

''' 
This is an API for the VL_Manager container to send information to the server
to run analysis using the ERMES package (which is installed in a different container). 
'''

def Run(AnalysisName, ContainerInfo=None, Append=False):
    if ContainerInfo is None:
        # Get default container info
        ContainerInfo = GetInfo('ERMES') 

    Wrapscript = "{}/ERMESExec.sh".format(Dir)
    command = "{} -c {} -f {} ".format(Wrapscript, ContainerInfo['Command'], AnalysisName)

    RC = Utils.Exec_Container(ContainerInfo, command)
    return RC
