import VLconfig

__all__ = ['ERMES']

ERMES = {'ContainerName':'SalomeMeca',
         'bind':[['/tmp'], 
                 ['/dev'], 
                 [VLconfig.VL_HOST_DIR,VLconfig.VL_DIR_CONT],
                ],
         'Command':"ERMESv12.5" 
        }

