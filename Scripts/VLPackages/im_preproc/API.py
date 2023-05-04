import os
import uuid
import pickle

from Scripts.Common.VLContainer import Container_Utils as Utils
from Scripts.VLPackages.ContainerInfo import GetInfo

Dir = os.path.dirname(os.path.abspath(__file__))

''' 
This is an API for the VL_Manager container to send information to the server
to the run image pre-processing package (which is installed in a different container). 
This is called in Methods/im_preproc.py
'''

def Run(funcfile, funcname, fnc_args=(), fnc_kwargs = {}, ContainerInfo = None, tempdir='/tmp'):
    
    if ContainerInfo is None:
        # Get default container info
        ContainerInfo = GetInfo('im_preproc') 

    pth = "{}/{}.pkl".format(tempdir,uuid.uuid4())
    with open(pth,'wb') as f:
        pickle.dump((fnc_args,fnc_kwargs),f)

    container_bash = "{}/VL_Preproc.sh".format(Dir) # bash script executed by container to perform setup ect.
    # command passed to container bash (done this way for more flexibility)
    container_command = "python3 /home/ibsim/VirtualLab/bin/run_pyfunc.py {} {} {}".format(funcfile,funcname,pth)

    command = "{} -c '{}' ".format(container_bash,container_command)
    RC = Utils.Exec_Container(ContainerInfo, command)
    return RC
