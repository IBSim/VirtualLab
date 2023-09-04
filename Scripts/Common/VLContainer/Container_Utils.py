import socket
import json
import pickle
import sys
import subprocess
import os
import uuid

def _tmpfile_pkl(tempdir="/tmp"):
    import uuid

    return "{}/{}.pkl".format(tempdir, uuid.uuid4())


def _pyfunctorun(funcfile, funcname, in_path, out_path):
    vlab_dir = get_vlab_dir()
    return "python3 {}/bin/run_pyfunc.py {} {} {} {}".format(
        vlab_dir, funcfile, funcname, in_path, out_path
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
    with open(ret_val_path, "rb") as f:
        func_results = pickle.load(f)

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

def get_Vlab_Host_Name():
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

    host_name = os.environ.get("VL_HOST_NAME", None)
    # set port number to the specified value then
    if host_name == None:
        print(
            "*************************************************************************\n",
            " WARNING: VL host name not found from environment variable $VL_HOST_NAME\n",
            "*************************************************************************",
        )
        sys.exit(1)
    return host_name

def run_pyfunc(ContainerInfo, funcfile, funcname, args=(), kwargs={}):
    python_exe, files = run_pyfunc_setup(funcfile, funcname, args=args, kwargs=kwargs)
    return run_pyfunc_launch(ContainerInfo, python_exe, files)


def bind_list2string(bind_list):
    """Returns a list of bind points in the format required by a container."""
    container_bind = []
    for bind_host,bind_cont in bind_list.items():
        container_bind.append(bind_host+":"+bind_cont)
    return ",".join(container_bind)

def bind_str2dict(bind_str):
    # comma separated list, with mount point denoted using :
    if len(bind_str)==0: return {}

    bind_dict = {}
    for _bind in bind_str.split(','):
        _bind_split = _bind.split(':')
        if len(_bind_split)==1:
            # same directory outside and inside
            host_path = cont_path = _bind_split[0]
        elif len(_bind_split)==2:
            host_path, cont_path = _bind_split
        else:
            print("****************************************************************")
            print("Warning: Unable to understand meaning of bind {}".format(_bind))
            print("****************************************************************")
        bind_dict[host_path] = cont_path
    return bind_dict

def path_change_binder(path, bindings, path_inside=True):
    """Converts path based on the bindings to the container used.
    This assumes that path is inside the container.
    Returns new path or None
    """

    for outside,inside in bindings.items():
        if path_inside:
            check_mount,swap_mount = inside, outside
        else:
            check_mount,swap_mount = outside, inside

        if path.startswith(check_mount):
            after_mount = path[len(check_mount) :]  # path after bind point
            swap_path = swap_mount + after_mount  # add this to swap_mount

            return swap_path

def is_bound(path,bind_dict):
    # return boolean value whether or not a certain path is in the container
    for host_path in bind_dict.keys():
        if path.startswith(host_path):
            return True
        
    return False


def Exec_Container(package_info, command):
    """Function called inside the VL_Manager container to pass information to VL_server
    to run jobs in other containers."""

    # Find out what stdout is to decide where to send output (for different modes).
    # This is updated on the server to give the filename on the host instead of the one inside VL_Manager

    sys.stdout.flush()  # flush information writen by manager to the file
    if sys.stdout.name == "<stdout>":
        stdout = None 
    else :
        stdout = sys.stdout.name

    # create new socket
    tcp_port = get_Vlab_Tcp_Port()
    host_name = socket.gethostname()
    sock = create_tcp_socket(host_name,tcp_port)

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



def MPI_Container(package_info, command, shared_dir,addpath=[],srun=False):
    """Function called inside the VL_Manager container to pass information to VL_server
    to run jobs in other containers."""

    # create new socket
    tcp_port = get_Vlab_Tcp_Port()
    host_name = socket.gethostname()
    sock = create_tcp_socket(host_name,tcp_port)

    # Create info dictionary to send to VLserver. The msg 'MPI' calls MPI_Container_Manager
    # on the server, where  'args' and 'kwargs' are passed to it.
    info = {
        "msg": "MPI",
        "Cont_id": 123,
        "Cont_name": package_info["ContainerName"],
        "shared_dir": shared_dir,
        "args": (package_info, command,shared_dir,tcp_port,host_name),
        "kwargs": {'addpath':addpath}
    }
    if srun:
        info['kwargs']['srun'] = True

    # send data to relevant function in VLserver
    send_data(sock, info)

    # Get the information returned by Exec_Container_Manager, which is the returncode of the subprocess
    ReturnCode = receive_data(sock, 0)  # return code from subprocess
    sock.close()  # cleanup after ourselves
    return ReturnCode

def _MPIFile(command,addpath):
    addpath_str = ":".join(addpath)
    info_list = ["#!/bin/bash",
                "export MPLBACKEND='Agg'",
                "export VL_TCP_PORT=$1",
                f"export PYTHONPATH={addpath_str}:$PYTHONPATH",
                command
                ] 
    string = "\n".join(info_list)
    return string

def MPI_Container_Manager(container_info, package_info, command, shared_dir, port, host_name,addpath=[],srun=False):
    """Function called on VL_server to run jobs on other containers."""
    
    container_cmd = container_info["container_cmd"]
    # merge in bind points from package and replace defaults
 
    container_info['bind'].update({'/dev':'/dev'})
    if package_info.get('bind',None) != None:
        container_info['bind'] = container_info['bind'] | package_info['bind']
    
    bind_str = bind_list2string(container_info["bind"])  # convert bind list to string
    container_cmd += " --bind {}".format(bind_str)  # update command with bind points
    #container_cmd += " --env PREPEND_PATH=/home/rhydian/VirtualLab/bin"
    _command = command.split()
    # command for running inside the container (to perform parallel evaluation of function)
    command_inside = _command[2:] if srun else _command[3:] # additional argument with srun to ignore
    command_inside = " ".join(command_inside) # command which will eb run inside the container

    # make file which initiates virtuallab requirememtns, e.g. conda environment
    contents = _MPIFile(command_inside,addpath)
    mpifile = "{}/MPIfile.sh".format(shared_dir)
    with open(mpifile,'w') as f:
        f.write(contents)

    # create command to launch container and execute mpifile
    run_container = [container_cmd,container_info["container_path"]] + [f'bash {mpifile}'] 
    run_container = " ".join(run_container)

    # command to be run on server
    if srun: launch_str = " ".join(_command[:2])
    else: launch_str = " ".join(_command[:3])

    this_dir = os.path.dirname(os.path.abspath(__file__))
    mpi_command = f"{launch_str} {this_dir}/MPI.sh '{run_container}' {host_name} {port} {shared_dir}"

    # run subprocess
    container_process = subprocess.Popen(mpi_command, shell=True)

    ReturnCode = (
        container_process.wait()
    )  # wait for process to finish and return its return code

    return ReturnCode

def create_tcp_socket(host, port_num):
    """Function to create the tcp socket and connect to it.
    """

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setblocking(True)
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



def build_container(Apptainer_file, container_loc):
    try:
        os.makedirs(os.path.dirname(Apptainer_file),exist_ok=True) # make sure the directory exists for the container to go into 
        proc = subprocess.check_call(
            f"apptainer build "
            f'{Apptainer_file} {container_loc}',
            shell=True,
        )
    except subprocess.CalledProcessError as E:
        print(E.stderr)
        raise E
    return

def get_container_path(Module):
    return f'docker://{Module["Docker_url"]}:{Module["Tag"]}'

def _upgrade_container(Apptainer_file,container_loc):

    basename,ext = os.path.splitext(Apptainer_file)
    random_suffix = str(uuid.uuid4())
    Apptainer_file_tmp = "{}_{}{}".format(basename,random_suffix,ext)

    ret = build_container(Apptainer_file_tmp,container_loc)
    os.remove(Apptainer_file)
    os.rename(Apptainer_file_tmp,Apptainer_file)

    return ret

def upgrade_container(ContainerName,Apptainer_file, container_loc):
    print(f'Building {ContainerName} container\n')
    if os.path.exists(Apptainer_file):
        return _upgrade_container(Apptainer_file,container_loc)
    elif not os.path.exists(Apptainer_file):
        return build_container(Apptainer_file, container_loc)
    
def check_container(ContainerName,Apptainer_file,container_loc):
    # check apptainer sif file exists and if not build from docker version
    if os.path.exists(Apptainer_file): return 

    print("Container doesn't seem to exist so building\n")
    print_container_info(ContainerName,Apptainer_file,container_loc)
    print('This may take a while\n')

    return build_container(Apptainer_file,container_loc)

def print_container_info(ContainerName,Apptainer_file,container_loc):
    print(f"Container name: {ContainerName}")
    print(f"Repo name: {container_loc}")
    print(f"Container location: {Apptainer_file}\n")




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

def container_to_host_path(file):
    # This function is no longer required as the host VL install is the same as that in the container
    return file