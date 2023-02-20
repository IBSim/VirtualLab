import VLconfig

__all__ = ['CodeAster','CodeAsterv14']

CodeAster = {'ContainerName':'SalomeMeca',
             'bind':[['/tmp'], 
                     ['/dev'], 
                     [VLconfig.VL_HOST_DIR,VLconfig.VL_DIR_CONT],
                    ],
             'Command':"/opt/SalomeMeca/V2019.0.3_universal/tools/Code_aster_frontend-20190/bin/as_run" 
            }
            
CodeAsterv14 = {'ContainerName':'Aster_test',
                'bind':[['/tmp'], 
                        ['/dev'], 
                        [VLconfig.VL_HOST_DIR,VLconfig.VL_DIR_CONT],
                       ],
                'Command':"/home/aster/aster/bin/as_run" 
               }            
            
            
            
