from ast import Raise
def Format_Call_Str(Tool,vlab_dir,param_master,param_var,Project,Simulation,use_singularity):
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
# Setup command to run inside container and bind directories based on tool used    
    if Tool == "CIL":
        if use_singularity:
            call_string = f'-B /run:/run -B {vlab_dir}:/home/ibsim/VirtualLab \
                            --nv Containers/CIL_sand'
        else:
            call_string = f'-v /run:/run -v {vlab_dir}:/home/ibsim/VirtualLab --gpus all ibsim/CIL'

        command = f'/home/ibsim/VirtualLab/Containers/Run_CIL.sh \
                   {param_master} {param_var} {Project} {Simulation}'

    elif Tool == "GVXR":
        if use_singularity:
            call_string = f'-B {vlab_dir}:/home/ibsim/VirtualLab --nv Containers/VL_GVXR.sif'
        else:
            call_string = f'-v {vlab_dir}:/home/ibsim/VirtualLab --gpus all ibsim/VL_GVXR'

        command = f'/home/ibsim/VirtualLab/Containers/Run_GVXR.sh \
                   {param_master} {param_var} {Project} {Simulation}'
    # Add other tools here as need arises
    else:
        Raise(ValueError("Tool not recognised as callable in container."))
    return call_string, command

def check_platform():
    '''Simple function to return True on Linux and false on Mac/Windows to
    allow the use of singularity instead of Docker on Linux systems.
    Singularity does not support Windows/Mac OS hence we need to check.
    Note: Docker can be used on Linux with the --docker flag. This flag
    however is ignored on both windows and Mac since they already
    default to Docker.'''
    import platform
    use_singularity=False
    if platform.system()=='Linux':
        use_singularity=True
    return use_singularity
