import VLconfig

__all__ = ['GVXR']

GVXR = {'ContainerName':'GVXR',
         'bind':[['/tmp'], 
                 ['/dev'], 
                 [VLconfig.VL_HOST_DIR,VLconfig.VL_DIR_CONT],
                ],
         'Command':"python" 
        }
        

