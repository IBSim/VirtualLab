import VLconfig

__all__ = ['cad2vox']

cad2vox = {'ContainerName':'Cad2Vox',
         'bind':[['/tmp'], 
                 ['/dev'], 
                 [VLconfig.VL_HOST_DIR,VLconfig.VL_DIR_CONT],
                ],
         'Command':"python" 
        }
        

