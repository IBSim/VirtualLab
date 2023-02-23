import VLconfig

__all__ = ['Salome']

Salome = {'ContainerName':'SalomeMeca',
         'bind':[['/tmp'], 
                 ['/dev'], 
                 [VLconfig.VL_HOST_DIR,VLconfig.VL_DIR_CONT],
                ],
         'Command':"salome" 
        }
        

