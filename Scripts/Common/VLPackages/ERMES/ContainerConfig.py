import VLconfig

__all__ = ['ERMES']

ERMES = {'ContainerFile':"/home/rhydian/VirtualLab/Containers/VL_Salome.sif",
         'bind':[['/tmp'], 
                 ['/dev'], 
                 [VLconfig.VL_HOST_DIR,VLconfig.VL_DIR_CONT],
                ],
         'Command':"ERMESv12.5" 
        }

