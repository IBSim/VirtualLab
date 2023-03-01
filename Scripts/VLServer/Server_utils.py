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
