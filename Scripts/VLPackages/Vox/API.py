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

def Run(funcfile, funcname, fnc_args=(), fnc_kwargs = {}, ContainerInfo = None, tempdir='/tmp'):
    
    if ContainerInfo is None:
        # Get default container info
        ContainerInfo = GetInfo('Vox') 

    # get python executable and temporary files created to run funcname as standalone
    python_exe, files = Utils.run_pyfunc_setup(funcfile,funcname,args=fnc_args,kwargs=fnc_kwargs)
    
    # need to set up certain parameters so create bash script (VL_Vox.sh) where python_exe is executed
    container_bash = "{}/VL_Vox.sh".format(Dir) 
    command = "{} -c '{}' ".format(container_bash,python_exe) # pass python_exe as argument to script 
    # run the above bash script. RC specifies whether the run was a success and func_return are the values returned by funcname
    RC, func_return = Utils.run_pyfunc_launch(ContainerInfo,command,files)    
 
    return RC
