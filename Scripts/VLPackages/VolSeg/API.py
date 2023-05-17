import os
import uuid
import pickle
import glob

from Scripts.Common.VLContainer import Container_Utils as Utils
from Scripts.VLPackages.ContainerInfo import GetInfo

Dir = os.path.dirname(os.path.abspath(__file__))

''' 
This is an API for the VL_Manager container to send information to the server
to run analysis using the Volume Segmentics package (which is installed in a different container). 
This is called in Methods/VolSeg.py
'''

def Run(mode :str, ContainerInfo = None,**kwargs):
    
    if ContainerInfo is None:
        # Get default container info
        ContainerInfo = GetInfo('VolSeg') 
    
    container_bash = "{}/VL_VolSeg.sh".format(Dir) # bash script executed by container
    if mode.lower() == 'train':
        command = f"{container_bash} -d {kwargs['Working_dir']} -c 'model-train-2d --data {kwargs['Samples']} --labels {kwargs['Labels']}'"
        print(command)
    elif mode.lower() == 'predict':
        model = kwargs.get('Model',None)
        if model == None:
            # FIX ME
            glob.glob(f"{kwargs['Working_dir']}/*.pytorch")
            print('hit')
        command = f"{container_bash} -d {kwargs['Working_dir']} -c 'model-predict-2d {model} {kwargs['Exp_Data']}'"
    else:
        raise ValueError('Invalid mode set for Volume Segmentics must be "predict" or "train"')
        
    RC = Utils.Exec_Container(ContainerInfo, command)
    return RC
