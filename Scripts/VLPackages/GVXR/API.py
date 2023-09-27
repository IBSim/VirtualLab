import os
import uuid
import pickle

from Scripts.Common.VLContainer import Container_Utils as Utils
from Scripts.VLPackages.ContainerInfo import GetInfo

Dir = os.path.dirname(os.path.abspath(__file__))

''' 
This is an API for the VL_Manager container to send information to the server
to run analysis using the cad2vox package (which is installed in a different container). 
This is called in Methods/Voxelise.py
'''

def Run(funcfile, funcname, fnc_args=(), fnc_kwargs = {}, ContainerInfo = None, return_values = True, tempdir='/tmp'):
    
    if ContainerInfo is None:
        # Get default container info
        ContainerInfo = GetInfo('GVXR') 

    python_exe = Utils.run_pyfunc_setup(funcfile,funcname,args=fnc_args,kwargs=fnc_kwargs)

    container_bash = "{}/VL_GVXR.sh".format(Dir) # bash script executed by container
    # command passed to container bash (done this way for more flexibility)
    command = "{} -c '{}' ".format(container_bash,python_exe)
    RC = Utils.Exec_Container(ContainerInfo, command)
    if return_values:
        func_return = Utils.run_pyfunc_return(python_exe)
        return RC, func_return
    else:
        return RC
