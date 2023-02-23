import VLconfig

__all__ = ['CIL']

CIL = {'ContainerName':'CIL',
         'bind':[['/tmp'], 
                 ['/dev'], 
                 [VLconfig.VL_HOST_DIR,VLconfig.VL_DIR_CONT],
                ],
         'Command':"python" 
        }
        

