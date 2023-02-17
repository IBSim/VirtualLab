import VLconfig

__all__ = ['Salome']

Salome = {'ContainerFile':"/home/rhydian/VirtualLab/Containers/VL_Salome.sif",
         'bind':[['/tmp'], 
                 ['/dev'], 
                 [VLconfig.VL_HOST_DIR,VLconfig.VL_DIR_CONT],
                ],
         'Command':"salome" 
        }
        

