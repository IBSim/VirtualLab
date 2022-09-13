from ast import Raise
def Format_Call_Str(Tool,vlab_dir,param_master,param_var,Project,Simulation):
    ''' Function to format string for bindpoints and container to call specified tool.'''
##### Format cmd argumants #########
    if param_var is None:
        param_var = ''
    else:
        param_var = '-v ' + param_var
    
    param_master = '-m '+ param_master
    Simulation = '-s ' + Simulation
    Project = '-p ' + Project
#########################################
# Format run string and script to run   #
# container based on tool used.         #
#########################################
    if Tool == "CIL":
        call_string = '-B /run:/run -B .:/home/ibsim/VirtualLab CIL_sand'
        command = '/home/ibsim/VirtualLab/Run_CIL.sh {} {} {} {}'.format(param_master,param_var,Project,Simulation)
        print(command)

    elif Tool == "GVXR":
        call_string = '-B {}:/home/ibsim/VirtualLab VL_GVXR.sif'.format(vlab_dir)
        command = '/home/ibsim/VirtualLab/Run_GVXR.sh {} {} {} {}'.format(param_master,param_var,Project,Simulation)
        
    # Add others as need arises
    else:
        Raise(ValueError("Tool not recognised as callable in container."))
    return call_string, command

def check_platform():
    '''Simple function to return True on Linux and false on Mac/Windows to
    automatically use singularity instead of Docker on Linux systems.
    Singularity does not support Windows/Mac OS hence we need to check.'''
    import platform
    use_singularity=False
    if platform.system()=='Linux':
        use_singularity=True
    return use_singularity