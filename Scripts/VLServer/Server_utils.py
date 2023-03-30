def ContainerError(out, err):
    """Custom function to format error message in a pretty way."""
    Errmsg = (
        "\n========= Container returned non-zero exit code =========\n\n"
        f"std out: {out}\n\n"
        f"std err:{err}\n\n"
        "==================================\n\n"
    )
    return Errmsg


def load_module_config_yaml(vlab_dir):
    """Function to get the config for the
    modules from VL_Modules.yaml file
    """
    # load module config from yaml_file
    config_file = vlab_dir / "Config/VL_Modules.yaml"
    with open(config_file) as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exception:
            print(exception)
    return config


def load_module_config(vlab_dir):
    """Function to get the config from a json file"""
    import json
    from pathlib import Path

    vlab_dir = Path(vlab_dir)
    # load module config from file
    config_file = vlab_dir / "Config/VL_Modules.json"
    with open(config_file) as file:
        config = json.load(file)
    return config


def correct_typecasting(arg):
    """
    This function is here to correct type casting mistakes made by the -k cmd option.
    The short version is arguments are passed in as strings but they may be ints
    floats or bools. This is particular problem for bool values as bool("False")
    evaluates as True in python. Thus this function exits to, hopefully, convert
    arguments to the expected/appropriate type.
    """
    # check for bool disguised as string
    if arg == "True":
        return True
    elif arg == "False":
        return False
    else:
        pass

    try:
        int(arg)
        return int(arg)
    except ValueError:
        # cant be cast as int but may be float
        pass
    # Check for float string
    try:
        float(arg)
        return float(arg)
    except ValueError:
        # assume arg is a string
        pass

    return arg


def check_file_in_container(vlab_dir, Run_file):
    """
    Function to check that the given runfile is accessible by the container i.e it is inside
    the virtualLab directory. If not the file is copied to the temporary directory, previously
    created by the tempfile library which is accessible and bound to /tmp in the container.
    """
    from pathlib import Path
    import shutil
    import os, sys, stat

    # Convert Run_file to a pathlib path
    Run_file = Path(Run_file)
    if not Run_file.is_absolute():
        Run_file = Run_file.resolve()

    if not Run_file.exists():
        raise ValueError(f"Runfile not found at {Run_file}.")
    return Run_file


def check_k_options(option):
    """
    check that the options given to -K are of the form Name=Value
    """
    import re

    # look for a single = sign that is not at the beging or end of the string.
    # Note: this coves a blank string and '='.
    matches = re.findall(r"\b=\b", option)
    if len(matches) != 1 or matches == []:
        print(
            f"invaid option {option} passed into -K this must be of the form Name=Value."
        )
        sys.exit(1)
    return


def check_valid_port(tcp_port):
    if tcp_port < 1024 or tcp_port > 65535:
        print(
            f"invalid port number {tcp_port} for option -P, must be an integer between 1024 and 65535."
        )
        sys.exit(0)
    else:
        return tcp_port

def host_to_container_path(filepath,vlab_dir_host,vlab_dir_cont = '/home/ibsim/VirtualLab'):
    """
    Function to Convert a path in the virtualLab directory on the host
    to an equivalent path inside the container. since the vlab _dir is
    mounted as /home/ibsim/VirtualLab inside the container.
    Note: The filepath needs to be absolute and is converted
    into a string before it is returned.
    """
    filepath=str(filepath)
    #check the path is accessible inside the container
    # that is it is relative to vlab_dir_host
    if filepath.startswith(str(vlab_dir_host)):
        # convert path to be relative to container not host
        filepath = str(filepath).replace(str(vlab_dir_host), str(vlab_dir_cont))
    elif filepath.startswith(vlab_dir_cont):
        # path is already relative to container so do nothing
        pass
    else:    
        raise FileNotFoundError(f"The path {filepath} is not accessible inside the Container.\n \
         The path must start with {vlab_dir_host}.")

    return filepath


def container_to_host_path(filepath,vlab_dir_host,vlab_dir_cont='/home/ibsim/VirtualLab'):
    """
    Function to Convert a path inside the container
    to an equivalent path on the host. since the vlab _dir is
    mounted as /home/ibsim/VirtualLab inside the container.

    Note: The filepath needs to be absolute and  is converted
    into a string before it is returned.
    """
    filepath=str(filepath)
    vlab_dir_host =str(vlab_dir_host)
    #check the path is accessible outside the container
    # that is it is relative to /home/ibsim/VirtualLab
    if filepath.startswith(str(vlab_dir_cont)):
        # convert path to be relative to host not container
        filepath = str(filepath).replace(str(vlab_dir_cont), str(vlab_dir_host))
    elif filepath.startswith(vlab_dir_host):
        # path is already relative to host so do nothing
        pass
    else:
        raise FileNotFoundError(f"The path {filepath} is not accessible outside the Container.\n \
        The path must start with {vlab_dir_cont}.")

    return filepath