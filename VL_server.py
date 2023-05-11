"""
        Script to enable communication with and spawning of containers.
        #######################################################################
        Note: Current containers are:
        1 - Base VirtualLab
        2 - CIL
        3 - GVXR
        4 - Container tests
        #######################################################################
"""
import socket
import subprocess
import threading
import argparse
import os
import sys
from pathlib import Path
import tempfile
sys.path.insert(0,os.path.dirname(Path(__file__).parent.resolve())) # add one level up

from Scripts.VLServer import Server_utils as Utils
from Scripts.Common.VLContainer.Container_Utils import (
    check_platform,
    send_data,
    receive_data,
    setup_networking_log,
    log_net_info,
    bind_list2string,
    bind_str2dict,
    path_change_binder,
    Exec_Container_Manager,
    MPI_Container_Manager,
    get_vlab_dir,
    update_container,
    is_bound,
    host_to_container_path,
)

vlab_dir = get_vlab_dir()
# do it this way as can't import VLconfig with VirtualLab binary
config_dict = Utils.filetodict("{}/VLconfig.py".format(vlab_dir))
ContainerDir = config_dict.get("ContainerDir", f"{vlab_dir}/Containers")

# global variables for use in all threads
waiting_cnt_sockets = {}
target_ids = []
task_dict = {}
settings_dict = {}
running_processes = {}
run_arg_dict = {}
Method_dict = {}
next_cnt_id = 1
manager_socket = None
cont_ready = False


##########################################################################################
####################  ACTUAL CODE STARTS HERE !!!! #######################################
##########################################################################################
def main():
    # read in CMD arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--Run_file",
        help="Where " "RUN_FILE" " is the path of the python run file.",
        default=None,
    )
    # parser.add_argument(
    #     "-D",
    #     "--Docker",VLconfig.VL_HOST_DIR
    # )
    parser.add_argument(
        "-X",
        "--debug",
        help="Flag to print debug messages for networking.",
        action="store_true",
    )
    parser.add_argument(
        "-T", "--test", help="Flag to initiate comms testing.", action="store_true"
    )
    parser.add_argument(
        "-N",
        "--no_nvidia",
        help="Flag to turn on/off nvidia support.",
        action="store_false",
    )
    parser.add_argument(
        "-d",
        "--dry_run",
        help="Flag to perform dry run.",
        action="store_true",
    )
    # parser.add_argument(
    #     "-C",
    #     "--nvccli",
    #     help="Flag to use nvidia continer toolkit instead of default --nv.",
    #     action="store_true",
    # )
    parser.add_argument(
        "-K",
        "--options",
        help="Overwrite the value specified for variables/keyword arguments specified in the Run file.",
        default=None,
        action="append",
        nargs="*",
    )
    parser.add_argument(
        "-P",
        "--tcp_port",
        help="tcp port to use for server communication. Default is 9000",
        default=9000,
        type=int,
    )
    parser.add_argument(
        "-B",
        "--bind",
        help="Additional files/directories to mount to containers.",
        default="",
    )
    # parser.add_argument(
    #     "-V",
    #     "--version",
    #     help="Display version number and citation information",
    #     action="store_false",
    # )

    # args = parser.parse_args()
    args, unknownargs = parser.parse_known_args()
    if unknownargs:
        sys.exit(f"Unknown option: {unknownargs[0]}")

    ################################################################
    # Note: Docker and nvcclli are work in progress options. As such
    # I don't want to totally remove them since they will be needed
    # if/when we fix the issues. However, I also don't want them to
    # appear as valid options with --help so when they are needed
    # simply delete/uncomment the appropriate lines.
    ###############################################################
    args.Docker = False
    args.nvccli = False
    ###############################################################

    # get vlab_dir either from cmd args or environment
    vlab_dir = get_vlab_dir()
    # Set flag to allow cmd switch between Apptainer and docker when using linux host.
    use_Apptainer = check_platform() and not args.Docker

    if len(sys.argv) == 1:
        if os.path.exists(f"{vlab_dir}/Citation.md"):
            with open(
                f"{vlab_dir}/Citation.md"
            ) as f:  # Write citation message to screen
                print(f.read())
        else:  # or write a more basic message as a backup if citation cant be found.
            print("***************************************")
            print("            VirtualLab-V2.0            ")
            print("A script to run VirtualLab simulations.")
            print("***************************************")
        # finally print usage to screen
        parser.print_help(sys.stderr)
        sys.exit(1)

    # set flag to run tests instate of the normal run file
    if args.test:
        Run_file = f"{vlab_dir}/RunFiles/Run_ComsTest.py"
    elif args.Run_file == None:
        print("****************************************************************")
        print("Error: you must specify a path to a valid RunFile with option -f")
        print("****************************************************************")
        parser.print_help(sys.stderr)
        sys.exit(1)
    else:
        Run_file = args.Run_file

    # ==========================================================================
    # make bindings to container

    # make a dir in /tmp on host with random name to avoid issues on shared systems
    # the tempfile library ensures this directory is deleted on exiting python.
    tmp_dir_obj = tempfile.TemporaryDirectory()
    tmp_dir = tmp_dir_obj.name

    bind_points_default = {
        "/usr/share/glvnd": "/usr/share/glvnd",
        str(Path.home()): str(Path.home()),
        str(tmp_dir): "/tmp",
        str(vlab_dir): "/home/ibsim/VirtualLab",
        "/dev": "/dev",
    }

    # add bind points given by command line
    _bind_dict = bind_str2dict(args.bind)
    for key, val in _bind_dict.items():
        if key in bind_points_default:
            continue
        bind_points_default[key] = val

    # add bind points defined in VLconfig
    _bind_points = config_dict.get("bind", "")
    _bind_dict = bind_str2dict(_bind_points)
    for key, val in _bind_dict.items():
        if key in bind_points_default:
            continue
        bind_points_default[key] = val

    # Add present working directory to the list of bind points if not already included
    pwd_dir = Utils.get_pwd()
    if not is_bound(pwd_dir, bind_points_default):
        bind_points_default[pwd_dir] = pwd_dir

    for dir_type in ["InputDir", "MaterialsDir", "OutputDir"]:
        _path = config_dict[dir_type]
        if not is_bound(_path, bind_points_default):
            message = (
                "\n*************************************************************************\n"
                f"Error: The '{dir_type}' directory '{_path}'\n"
                "is not bound to the container. This can be corrected either using the \n"
                "--bind option or by including the argument bind in VLconfig.py\n"
                "*************************************************************************\n"
            )

            sys.exit(message)
    # output final bind point ict to file so we can retrieve them later
    Utils.bind_points2file(bind_points_default, vlab_dir)
    ######################################
    # formatting for optional -K cmd option
    kOption_dict = {}
    if args.options != None:
        options = ""
        # Note: -K can be set multiple times so we need these loops to format them correctly to be passed on
        for N, opt in enumerate(args.options):
            for n, _ in enumerate(opt):
                Utils.check_k_options(opt[n])
                options = options + " -k " + opt[n]
                key = opt[n].split("=")[0]
                value = opt[n].split("=")[1]
                kOption_dict[key] = value
    else:
        options = ""

    ####################################################
    # pass debug and dry_run flags in as k options if set
    ####################################################
    if args.dry_run:
        options = options + " -k dry_run=True"
    if args.debug:
        options = options + " -k debug=True"
    if args.tcp_port:
        tcp_port = Utils.check_valid_port(args.tcp_port)
        options = options + f" -P {tcp_port}"

    #####################################
    # turn on/off gpu support with a flag
    gpu_support = args.no_nvidia
    if gpu_support and args.nvccli:
        gpu_flag = "--nvccli"
    elif gpu_support:
        gpu_flag = "--nv"
    else:
        gpu_flag = ""

        print("##############################################")
        print("VirtualLab Running in software rendering mode.")
        print("##############################################")
    # test never needs gpu support
    if args.test:
        gpu_flag = ""
    # start server listening for incoming jobs on separate thread
    lock = threading.Lock()
    thread = threading.Thread(
        target=process,
        args=(
            vlab_dir,
            use_Apptainer,
            args.debug,
            gpu_flag,
            tcp_port,
            bind_points_default,
        ),
    )
    thread.daemon = True

    Modules = Utils.load_module_config(vlab_dir)
    Manager = Modules["Manager"]

    if Run_file.startswith("/") and not is_bound(Run_file, bind_points_default):
            message = (
                "\n*************************************************************************\n"
                f"Error: The Runfile {Run_file}\n" \
                "is in a directory that is not bound to the container. This can be corrected\n" \
                "either using the --bind option or by including the argument bind in VLconfig.py\n"
                "*************************************************************************\n"
            )
            sys.exit(message)

    # convert path from host to container if needed
    path = host_to_container_path(Run_file, vlab_dir)
    thread.start()
    # start VirtualLab
    lock.acquire()

    # convert default bind points to container style string
    bind_str = bind_list2string(bind_points_default)

    if use_Apptainer:
        if Manager["Apptainer_file"].startswith("/"):
            # full path provided
            Apptainer_file = Manager["Apptainer_file"]
        else:
            # relative path from container dir
            Apptainer_file = f"{ContainerDir}/{Manager['Apptainer_file']}"

        if not os.path.exists(Apptainer_file):
            update_container(Manager, vlab_dir)

        proc = subprocess.Popen(
            f"apptainer exec --contain --writable-tmpfs \
                        --bind {bind_str} {Apptainer_file} "
            f'{Manager["Startup_cmd"]} {options} -f {path} ',
            shell=True,
        )
    else:
        # Assume using Docker
        proc = subprocess.Popen(
            f"docker run --rm -it --network=host -v {vlab_dir}:/home/ibsim/VirtualLab "
            f'{Manager["Docker_url"]}:{Manager["Tag"]} '
            f'"{Manager["Startup_cmd"]} {options} -f {Run_file}"',
            shell=True,
        )
    lock.release()

    # wait until virtualLab is done before closing
    err = proc.wait()
    sys.exit(err)


def handle_messages(
    client_socket,
    net_logger,
    VL_MOD,
    sock_lock,
    cont_ready,
    debug,
    gpu_flag,
    bind_points_default,
):
    global waiting_cnt_sockets
    global target_ids
    global task_dict
    global settings_dict
    global running_processes
    global next_cnt_id
    global manager_socket
    global run_arg_dict
    global Method_dict
    # list of messages to simply relay from Container_id to Target_id
    relay_list = ["Continue", "Waiting", "Error"]
    while True:
        rec_dict = receive_data(client_socket, debug)

        if rec_dict == None:
            log_net_info(net_logger, "Socket has been closed")
            return
        event = rec_dict["msg"]

        container_id = rec_dict["Cont_id"]
        log_net_info(
            net_logger,
            f'Server - received "{event}" event from container {container_id}',
        )

        pwd_dir = Utils.get_pwd()
        if event in ("Exec","MPI"):
            # will need to add option for docker when fixed

            container_cmd = (
                f"apptainer exec --contain {gpu_flag} --writable-tmpfs -H {pwd_dir}"
            )

            cont_name = rec_dict["Cont_name"]
            cont_info = VL_MOD[cont_name]

            if cont_info["Apptainer_file"].startswith("/"):
                container_path = cont_info["Apptainer_file"]  # full path provided
            else:
                container_path = f"{ContainerDir}/{cont_info['Apptainer_file']}"  # relative path from container dir

            cont_info["container_path"] = container_path
            cont_info["container_cmd"] = container_cmd
            cont_info["bind"] = bind_points_default

            # check apptainer sif file exists and if not build from docker version
            if not os.path.exists(container_path):
                os.makedirs(os.path.dirname(container_path), exist_ok=True)
                # sif file doesn't exist
                if "Docker_url" in cont_info:
                    print(
                        f"Apptainer file {container_path} does not appear to exist so building. This may take a while."
                    )
                    try:
                        proc = subprocess.check_call(
                            f"apptainer build "
                            f'{container_path} docker://{cont_info["Docker_url"]}:{cont_info["Tag"]}',
                            shell=True,
                        )
                    except subprocess.CalledProcessError as E:
                        print(E.stderr)
                        raise E

                else:
                    print(
                        f"Apptainer file {container_path} does not exist and no information about its location is provided.\n Exiting"
                    )
                    sys.exit()

            args = rec_dict.get("args", ())
            kwargs = rec_dict.get("kwargs", {})

            if event == "Exec":
                stdout = kwargs.get("stdout", None)
                if stdout is not None and not os.path.isdir(os.path.dirname(stdout)):
                    # stdout is a file path within VL_Manager so need to get the path on the host
                    stdout = path_change_binder(stdout, bind_points_default)
                    kwargs["stdout"] = stdout

                RC = Exec_Container_Manager(cont_info, *args, **kwargs)
                send_data(client_socket, RC, debug)
            else:
                # MPI
                RC = MPI_Container_Manager(cont_info, *args, **kwargs)
                send_data(client_socket, RC, debug)

        elif event == "Build":
            # recive list of containers to be built
            cont_names = rec_dict["Cont_names"]
            dont_exist = []
            for Cont in cont_names:
                print(f"Build {Cont}")
                Module = VL_MOD.get(Cont, None)
                if Module == None:
                    # add to list of containers not found.
                    dont_exist.append(Cont)
                else:
                    update_container(Module, vlab_dir)
            if dont_exist == []:
                message = {"msg": "Done Building"}
            else:
                message = {"msg": "Build Error", "Cont_names": dont_exist}
            send_data(client_socket, message)

        elif event in relay_list:
            Target_id = str(rec_dict["Target_id"])
            Target_socket = waiting_cnt_sockets[Target_id]
            send_data(Target_socket, rec_dict, debug)
        else:
            client_socket.shutdown(socket.SHUT_RDWR)
            client_socket.close()
            raise ValueError(f"Unknown message {event} received")


def process(vlab_dir, use_Apptainer, debug, gpu_flag, tcp_port, bind_points_default):
    """Function that runs in a thread to handle communication ect."""
    global waiting_cnt_sockets
    next_cnt_id = 1
    global manager_socket
    cont_ready = threading.Event()
    log_dir = f"{vlab_dir}/.log/network_log"
    net_logger = setup_networking_log(log_dir)
    sock_lock = threading.Lock()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setblocking(True)
    host = "0.0.0.0"
    sock.bind((host, tcp_port))
    sock.listen(20)
    VL_MOD = Utils.load_module_config(vlab_dir)

    ################################
    while True:
        # check for new connections and them to list
        client_socket, client_address = sock.accept()

        waiting_cnt_sockets[str(next_cnt_id)] = {
            "socket": client_socket,
            "id": next_cnt_id,
        }
        next_cnt_id += 1
        # spawn a new thread to deal with messages
        thread = threading.Thread(
            target=handle_messages,
            args=(
                client_socket,
                net_logger,
                VL_MOD,
                sock_lock,
                cont_ready,
                debug,
                gpu_flag,
                bind_points_default,
            ),
        )
        thread.daemon = True
        thread.start()


if __name__ == "__main__":
    main()
