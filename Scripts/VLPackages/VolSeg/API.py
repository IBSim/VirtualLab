import os
import uuid
import pickle

from Scripts.Common.VLContainer import Container_Utils as Utils
from Scripts.VLPackages.ContainerInfo import GetInfo

Dir = os.path.dirname(os.path.abspath(__file__))

''' 
This is an API for the VL_Manager container to send information to the server
to run analysis using the Survos package (which is installed in a different container). 
This is called in Methods/Survos.py
'''

def Run(ContainerInfo = None, tempdir='/tmp',**kwargs):
    
    if ContainerInfo is None:
        # Get default container info
        ContainerInfo = GetInfo('Survos') 

    container_bash = "{}/VL_Survos.sh".format(Dir) # bash script executed by container
    # command passed to container bash (done this way for more flexibility)
    container_command = ContainerInfo['Command']

    command = "{} -c '{}' ".format(container_bash,container_command)
    RC = Utils.Exec_Container(ContainerInfo, command)
    return RC
