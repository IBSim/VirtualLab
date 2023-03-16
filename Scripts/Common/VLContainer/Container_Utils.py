from email import message
import socket
import json
import tempfile
import pickle

from types import SimpleNamespace as Namespace
from ast import Raise
import struct
import sys
import subprocess


def _tmpfile_pkl(tempdir="/tmp"):
    import uuid

    return "{}/{}.pkl".format(tempdir, uuid.uuid4())


def _pyfunctorun(funcfile, funcname, in_path, out_path):
    return "python3 /home/ibsim/VirtualLab/bin/run_pyfunc.py {} {} {} {}".format(
        funcfile, funcname, in_path, out_path
    )


def run_pyfunc_setup(funcfile, funcname, args=(), kwargs={}):
    arg_path = _tmpfile_pkl()  # temp file for arguments
    with open(arg_path, "wb") as f:
        pickle.dump((args, kwargs), f)

    ret_val_path = _tmpfile_pkl()  # temp file for return of function

    python_exe = _pyfunctorun(funcfile, funcname, arg_path, ret_val_path)

    return python_exe, [arg_path, ret_val_path]


def run_pyfunc_launch(ContainerInfo, command, pkl_files):
    RC = Exec_Container(ContainerInfo, command)
    arg_path, ret_val_path = pkl_files
    if RC == 0:
        with open(ret_val_path, "rb") as f:
            func_results = pickle.load(f)
    else:
        #an error occurred so there will not be any func_return
        func_results= None

    return RC, func_results


def get_Vlab_Tcp_Port():
    """
    Function to get vlab tcp port from the os environment.
    variable VL_TCP_PORT. This variable is, or at least
    should be set in the VL_Manager container when VL_setup
    is first created.

    This then allows you to easily create new tcp sockets
    for spawning containers without having to pass objects
    through the various layers of functions.

    WARNING: this function should only be called inside the
    VL_Manager container otherwise it will not work
    as the environment variables will be different to other
    containers or the host.
    """
    import os

    port_num = os.environ.get("VL_TCP_PORT", None)
    # set port number to the specified value then
    if port_num == None:
        print(
            "*************************************************************************\n",
            " WARNING: TCP Port number not found from environment variable $VL_TCP_PORT\n",
            " This should not happen unless you are either:\n",
            " A) Calling this function outside of the VL_Manager Container\n",
            " B) you have somehow not called _SetTcp_Port during the Settings function of VLSetup.\n",
            "*************************************************************************",
        )
        sys.exit(1)
    return int(port_num)


def run_pyfunc(ContainerInfo, funcfile, funcname, args=(), kwargs={}):
    python_exe, files = run_pyfunc_setup(funcfile, funcname, args=args, kwargs=kwargs)
    return run_pyfunc_launch(ContainerInfo, python_exe, files, sock)


def bind_list2string(bind_list):
    """Returns a list of bind points in the format required by a container."""
    container_bind = []
    for bind_host,bind_cont in bind_list.items():
        container_bind.append(bind_host+":"+bind_cont)
    return ",".join(container_bind)


def path_change_binder(path, bind_list, path_inside=True):
    """Converts path based on the bindings to the container used.
    This assumes that path is inside the container.
    Returns new path or None
    """
    if path_inside:
        check_ix, swap_ix = 1, 0
    else:
        check_ix, swap_ix = 0, 1

    for bind in bind_list:
        if len(bind) == 1:
            bind = bind * 2  # make in to two

        check_mount, swap_mount = bind[check_ix], bind[swap_ix]
        if path.startswith(check_mount):
            after_mount = path[len(check_mount) :]  # path after bind point
            swap_path = swap_mount + after_mount  # add this to swap_mount

            return swap_path


def Exec_Container(package_info, command):
    """Function called inside the VL_Manager container to pass information to VL_server
    to run jobs in other containers."""

    # Find out what stdout is to decide where to send output (for different modes).
    # This is updated on the server to give the filename on the host instead of the one inside VL_Manager
    stdout = None if sys.stdout.name == "<stdout>" else sys.stdout.name

    # create new socket
    tcp_port = get_Vlab_Tcp_Port()
    sock = create_tcp_socket(tcp_port)

    # Create info dictionary to send to VLserver. The msg 'Exec' calls Exec_Container_Manager
    # on the server, where  'args' and 'kwargs' are passed to it.
    info = {
        "msg": "Exec",
        "Cont_id": 123,
        "Cont_name": package_info["ContainerName"],
        "args": (package_info, command),
        "kwargs": {"stdout": stdout},
    }

    # send data to relevant function in VLserver
    send_data(sock, info)

    # Get the information returned by Exec_Container_Manager, which is the returncode of the subprocess
    ReturnCode = receive_data(sock, 0)  # return code from subprocess
    sock.close()  # cleanup after ourselves
    return ReturnCode


def Exec_Container_Manager(container_info, package_info, command, stdout=None):
    """Function called on VL_server to run jobs on other containers."""
    
    container_cmd = container_info["container_cmd"]
    # merge in bind points from package and replace defaults

    if package_info.get('bind',None) != None:
        container_info['bind'] = container_info['bind'] | package_info['bind']
    bind_str = bind_list2string(container_info["bind"])  # convert bind list to string
    container_cmd += " --bind {}".format(bind_str)  # update command with bind points

    # SP_call is whats executed by the server. calls containers and passes commands to it
    SP_call = "{} {} {}".format(
        container_cmd, container_info["container_path"], command
    )

    if stdout is None:
        # output just goes to stdout
        container_process = subprocess.Popen(SP_call, shell=True)
    else:
        # output gets written to file instead
        with open(stdout, "a") as outhandle:
            container_process = subprocess.Popen(
                SP_call, shell=True, stdout=outhandle, stderr=outhandle
            )

    ReturnCode = (
        container_process.wait()
    )  # wait for process to finish and return its return code
    return ReturnCode



def create_tcp_socket(port_num=9000):
    """Function to create the tcp socket and connect to it.
    The default port is 9000 for coms with the containers.
    """

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setblocking(True)
    host = "0.0.0.0"
    sock.connect((host, port_num))
    return sock



def send_data(conn, payload, bigPayload=False, debug=False):
    """
    Adapted from: https://github.com/vijendra1125/Python-Socket-Programming/blob/master/server.py
    @brief: send payload along with data size and data identifier to the connection
    @args[in]:
        conn: socket object for connection to which data is supposed to be sent
        payload: payload to be sent
        bigFile: flag to suppress warning if payload is larger than the standard 2048 bytes.

        This is here because the dict is dynamically generated at runtime and may become larger
        than the default buffer without you necessarily knowing about it.

        There is no reason you cant send larger data than this (see payload_size argument
        for receive_data bellow).

        The warning is merely here to save you from yourself and allow you to make
        adjustments to avoid errors caused by data overflows.
    """
    # serialize payload

    if debug:
        print(f"sent:{payload}")
    serialized_payload = json.dumps(payload).encode("utf-8")
    payload_size = len(serialized_payload)
    if payload_size > 2048 and not bigPayload:
        print(
            "###################################################\n"
            f"Warning: Payload has a size of {payload_size} bytes.\n"
            "This exceeds the standard buffer size of 2048 bytes.\n"
            "You will need to ensure you set the buffer on the \n"
            "corresponding call to receive_data to a large \n"
            "enough value or else data may be lost/corrupted.\n"
            "To suppress this message set the bigPayload flag.\n"
            "###################################################"
        )
    conn.sendall(serialized_payload)


def receive_data(conn, debug=False, payload_size=2048):
    """
    @brief: receive data from the connection assuming that data is a json string
    @args[in]:
        conn: socket object for connection from which data is supposed to be received
        payload_size: size in bytes of the object buffer for the TCP protocol.

        Note this is not the size of the object itself, that can be much smaller.
        This number is the amount of memory allocated to hold the received object. It must
        therefore be large enough to hold the object. For now this is set to an ample
        default of 2Kb. However, since the dicts are generated dynamically at run time
        they may become larger than this. If so just set this to a large enough number.

        You may also want to set the bigPayload flag in send_data.

    """
    received_payload = conn.recv(payload_size)

    if not received_payload:
        payload = None
    else:
        received_payload = received_payload.decode("utf-8")
        payload = json.loads(received_payload)
        if debug:
            print(f"received:{payload}")
    return payload


def Format_Call_Str(Module, vlab_dir, class_file, pythonpaths, use_Apptainer, cont_id):
    """Function to format string for bind points and container to call specified tool."""
    import os
    import subprocess

    ##### Format cmd argumants #########
    filepath = "-m " + class_file
    ID = "-I " + str(cont_id)
    pypath = "-p " + ":".join(pythonpaths)

    #########################################
    # Format run string and script to run   #
    # container based on Module used.       #
    #########################################
    if use_Apptainer:
        import random

        update_container(Module, vlab_dir)
        call_string = f' -B /run:/run -B /tmp:/tmp --contain -B {str(vlab_dir)}:/home/ibsim/VirtualLab \
                        {str(vlab_dir)}/{Module["Apptainer_file"]} '
    else:
        # docker
        call_string = f'-v /run:/run -v /tmp:/tmp -v {str(vlab_dir)}:/home/ibsim/VirtualLab {Module["Docker_url"]}:{Module["Tag"]}'

    # get custom command line arguments if specified in config.
    arguments = Module.get("cmd_args", None)
    if arguments == None:
        command = f'{Module["Startup_cmd"]} \
               {filepath} {ID} {pypath}'
    else:
        command = f'{Module["Startup_cmd"]} {arguments}'

    return call_string, command


def check_platform():
    """Simple function to return True on Linux and false on Mac/Windows to
    allow the use of Apptainer instead of Docker on Linux systems.
    Apptainer does not support Windows/Mac OS hence we need to check.
    Note: Docker can be used on Linux with the --docker flag. This flag
    however is ignored on both windows and Mac since they already
    default to Docker."""
    import platform

    use_Apptainer = False
    if platform.system() == "Linux":
        use_Apptainer = True
    return use_Apptainer


def setup_networking_log(filename):
    """
    Setup two loggers one for file and one for the screen.
    The file logger is set to debug so it should catch
    anything sent for logging. The screen is set to Info
    so it will display anything that is not marked debug.

    For reference Log levels are:
    DEBUG
    INFO
    WARNING
    ERROR
    CRITICAL
    """
    import logging
    from logging.handlers import TimedRotatingFileHandler
    import datetime

    now = datetime.datetime.now()
    today = now.strftime("%Y-%m-%d")
    filename = f"{filename}_{today}.log"
    log = logging.getLogger("logger")
    # Sets the base level for all logging.
    # Setting this to debug ensures we log everything.
    # Since default level is Warning if we didn't
    # set this and used debug in one of our handlers
    # it wouldn't log anything below warning.
    log.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(message)s")

    # Logger for file
    fh = logging.FileHandler(filename, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)

    # Logger for screen
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    log.addHandler(ch)

    # print date and time to log for starting virtualLab

    log.debug(f"started VirtualLab:{now}")
    return log


def log_net_info(logger, message, screen=False):
    if screen:
        logger.info(message)
    else:
        logger.debug(message)


def update_container(Module, vlab_dir):
    import os
    import subprocess

    Apptainer_file = f"{vlab_dir}/{Module['Apptainer_file']}"
    # check apptainer sif file exists and if not build from docker version
    if not os.path.exists(Apptainer_file):
        print(
            f"Apptainer file {Apptainer_file} does not appear to exist so building. This may take a while."
        )
        try:
            proc = subprocess.check_call(
                f"apptainer build "
                f'{Apptainer_file} docker://{Module["Docker_url"]}:{Module["Tag"]}',
                shell=True,
            )
        except subprocess.CalledProcessError as E:
            print(E.stderr)
            raise E
    return


def get_vlab_dir(parsed_dir=None):
    """
    Function to get path to vlab_dir from either:
    input function parameters or os environment. in that order.
    If nether is possible it defaults to the users home directory.
    which will be either /home/{user}/VirtualLab
    or C:\Documents\VirtualLab depending upon the OS.

    If the given directory does not exist it raises a value error.

    """
    import os
    from pathlib import Path

    if parsed_dir != None:
        vlab_dir = Path(parsed_dir)
        os.environ["VL_DIR"] = str(parsed_dir)
    else:
        # get dir from OS environment which should be set during installation
        vlab_dir = os.environ.get("VL_DIR", None)
        if vlab_dir == None:
            vlab_dir = Path.home() / "VirtualLab"
        else:
            # here because you can't create a Path object from None
            vlab_dir = Path(vlab_dir)

    if not vlab_dir.is_dir():
        raise ValueError(
            f"Could not find VirtualLab install directory. The directory {str(vlab_dir)} does not appear to exist. \n"
            " Please specify where to find the VirtualLab install directory by setting the environment variable VL_DIR."
        )

    return vlab_dir


def host_to_container_path(filepath):
    """
    Function to Convert a path in the virtualLab directory on the host
    to an equivalent path inside the container. since the vlab _dir is
    mounted as /home/ibsim/VirtualLab inside the container.
    Note: The filepath needs to be absolute and is converted
    into a string before it is returned.
    """
    import VLconfig as VLC
    filepath=str(filepath)
    vlab_dir_host = VLC.VL_HOST_DIR
    # location of vlab inside the container
    cont_vlab_dir = VLC.VL_DIR_CONT
    #check the path is accessible inside the container
    # that is it is relative to VLC.VL_HOST_DIR
    if filepath.startswith(vlab_dir_host):
        # convert path to be relative to container not host
        filepath = str(filepath).replace(str(vlab_dir_host), str(cont_vlab_dir))
    elif filepath.startswith(cont_vlab_dir):
        # path is already relative to container so do nothing
        pass
    else:    
        raise FileNotFoundError(f"The path {filepath} is not accessible inside the Container.\n \
         The path must start with {VLC.VL_HOST_DIR}.")

    return filepath


def container_to_host_path(filepath):
    """
    Function to Convert a path inside the container
    to an equivalent path on the host. since the vlab _dir is
    mounted as /home/ibsim/VirtualLab inside the container.

    Note: The filepath needs to be absolute and  is converted
    into a string before it is returned.
    """
    import VLconfig as VLC
    filepath=str(filepath)
    vlab_dir_host = VLC.VL_HOST_DIR    
    # location of vlab inside the container
    cont_vlab_dir = VLC.VL_DIR_CONT
    #check the path is accessible outside the container
    # that is it is relative to VLC.VL_DIR_CONT
    if filepath.startswith(cont_vlab_dir):
        # convert path to be relative to host not container
        filepath = str(filepath).replace(str(cont_vlab_dir), str(vlab_dir_host))
    elif filepath.startswith(vlab_dir_host):
        # path is already relative to host so do nothing
        pass
    else:
        raise FileNotFoundError(f"The path {filepath} is not accessible outside the Container.\n \
        The path must start with {VLC.VL_DIR_CONT}.")

    return filepath
