import VLconfig

__all__ = ['CodeAster']

CodeAster = {'ContainerFile':"/home/rhydian/VirtualLab/Containers/VL_Salome.sif",
             'bind':[['/tmp'], 
                     ['/dev'], 
                     [VLconfig.VL_HOST_DIR,VLconfig.VL_DIR_CONT],
                     ['/home/rhydian/flasheur','/home/ibsim/flasheur']
                    ],
             'Command':"/opt/SalomeMeca/V2019.0.3_universal/tools/Code_aster_frontend-20190/bin/as_run" 
            }
