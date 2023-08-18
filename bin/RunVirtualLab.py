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
import pickle

import Scripts.VLServer.Server_utils as SU
import Scripts.Common.VLContainer.Container_Utils as CU

vlab_dir = CU.get_vlab_dir()
# do it this way as can't import VLconfig with VirtualLab binary
config_dict = SU.filetodict("{}/VLconfig.py".format(vlab_dir))
ContainerDir = config_dict.get('ContainerDir',f"{vlab_dir}/Containers")

VL_MOD = SU.load_module_config(vlab_dir)

##########################################################################################
####################  ACTUAL CODE STARTS HERE !!!! #######################################
##########################################################################################

def main():    
    routine = sys.argv[1] if len(sys.argv)>1 else ''

    if routine == 'server_start':
        start_server(*sys.argv[2:4])
    elif routine == 'server_kill':
        kill_server(sys.argv[2])
    elif routine == 'hostname':
        get_host(sys.argv[2])
    else:
        runVL()

def runVL():
    # read in CMD arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--Run_file",
        help="Where " "RUN_FILE" " is the path of the python run file.",
        default=None,
    )
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
        help="tcp port to use for server communication.",
        default=None,
        type=int,
    )
    parser.add_argument(
        "-B",
        "--bind",
        help="Additional files/directories to mount to containers.",
        default='',
    )
    parser.add_argument(
        "-U",
        "--upgrade_container",
        help="Force containers to be pulled to get newest versions.",
        default=''
    )           
    # parser.add_argument(
    #     "-C",
    #     "--nvccli",
    #     help="Flag to use nvidia continer toolkit instead of default --nv.",
    #     action="store_true",
    # )
     # parser.add_argument(
    #     "-D",
    #     "--Docker",VLconfig.VL_HOST_DIR
    # )
    # parser.add_argument(
    #     "-V",
    #     "--version",
    #     help="Display version number and citation information",
    #     action="store_false",
    # )

    ##############################################################
    # check for any unknown arguments parsed
    args, unknownargs = parser.parse_known_args()
    if unknownargs:
       print(f'Unknown option: {unknownargs[0]}\n')
       parser.print_help(sys.stderr)
       sys.exit(1)

    ###############################################################
    # upgrade containers if flag used
    if args.upgrade_container: # string is not empty
        _upgrade_container(args.upgrade_container)


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
    else:
        Run_file = args.Run_file

    if Run_file:
        _run_file(Run_file,args)

def _run_file(Run_file,args):
    '''
    Function which spins up server to communicate with the run file to use other containers
    '''
    
    Run_file = SU.check_file_in_container(vlab_dir, Run_file)

    # Set flag to allow cmd switch between Apptainer and docker when using linux host.
    use_Apptainer = CU.check_platform() and not args.Docker

    # ==========================================================================
    # make bindings to container

    # make a dir in /tmp on host with random name to avoid issues on shared systems
    # the tempfile library ensures this directory is deleted on exiting python.
    # tmp_dir_obj = tempfile.TemporaryDirectory()
    # tmp_dir = tmp_dir_obj.name
    tmp_dir='/tmp'
    # default bind points used in every container
    bind_points_default = { "/usr/share/glvnd":"/usr/share/glvnd",
                            str(Path.home()):str(Path.home()),
                            str(tmp_dir):"/tmp",
                            str(vlab_dir):"/home/ibsim/VirtualLab",
                          }

    # bind points defined in VLconfig
    _bind_points = config_dict.get('bind','')
    _bind_dict_add = CU.bind_str2dict(_bind_points)

    # add bind points given by command line
    _bind_dict = CU.bind_str2dict(args.bind)
    _bind_dict_add.update(_bind_dict)
    for key,val in _bind_dict_add.items():
        if key in bind_points_default: continue
        bind_points_default[key] = val

    # Add present working directory to the list of bind points if not already included
    pwd_dir = SU.get_pwd()
    if not CU.is_bound(pwd_dir,bind_points_default):
        bind_points_default[pwd_dir] = pwd_dir


    for dir_type in ['InputDir','MaterialsDir','OutputDir']:
        _path = config_dict[dir_type]
        if not CU.is_bound(_path,bind_points_default):
            message = "\n*************************************************************************\n" \
            f"Error: The '{dir_type}' directory '{_path}'\n" \
            "is not bound to the container. This can be corrected either using the \n" \
            "--bind option or by including the argument bind in VLconfig.py\n" \
            "*************************************************************************\n"
            
            sys.exit(message)
    
    ######################################
    # formatting for optional -K cmd option
    kOption_dict = {}
    if args.options != None:
        options = ""
        # Note: -K can be set multiple times so we need these loops to format them correctly to be passed on
        for N, opt in enumerate(args.options):
            for n, _ in enumerate(opt):
                SU.check_k_options(opt[n])
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


    # make socket on a port
    host = socket.gethostname()
    if args.tcp_port:
        # use given port number
        tcp_port = SU.check_valid_port(args.tcp_port)
        sock = make_socket(host,tcp_port)
    else:
        # run on free port
        sock = make_socket(host)
        tcp_port = sock.getsockname()[1]
   
    options = options + f" -P {tcp_port} -s {host}"

    # start server listening for incoming jobs on separate thread
    lock = threading.Lock()
    thread = threading.Thread(
        target=process,
        args=(vlab_dir, sock, args, gpu_flag, bind_points_default),
    )

    thread.daemon = True

    Modules = SU.load_module_config(vlab_dir)
    Manager = Modules["Manager"]

    thread.start()
    # start VirtualLab
    lock.acquire()

    # convert default bind points to container style string
    bind_str = CU.bind_list2string(bind_points_default)

    if use_Apptainer:
        Apptainer_file = _ContainerFull(Manager['Apptainer_file'])
        container_loc = CU.get_container_path(Manager)
        CU.check_container('Manager', Apptainer_file, container_loc)

        proc = subprocess.Popen(
                f"apptainer exec -H {pwd_dir} --contain --writable-tmpfs \
                                 --bind {bind_str} {Apptainer_file} "
                f'/home/ibsim/VirtualLab/bin/VL_Manager {options} -f {Run_file} ',
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

def _ContainerFull(ApptainerFile):
    if not ApptainerFile.startswith('/'): # relative path from container dir
        ApptainerFile = f"{ContainerDir}/{ApptainerFile}"
    return ApptainerFile

def _upgrade_container(container_str):

    if container_str=='current':
        container_list = []
        for container_name,container_info in VL_MOD.items():
            Apptainer_file = _ContainerFull(container_info['Apptainer_file'])
            if os.path.exists(Apptainer_file):
                container_list.append(container_name) # add files which have already been downloaded
    elif container_str=='all':
        container_list = list(VL_MOD.keys())
    else:
        _container_list = container_str.split(',') # download named files (comma separated)
        container_list = []
        for container_name in _container_list:
            # check the names given are available
            if container_name not in VL_MOD:
                print(f'\nError: Container {container_name} is not available to be upgraded\n')
            else:
                container_list.append(container_name)

    if len(container_list)==0: return

    print('The following containers will be upgraded:\n')
    info = []
    for container_name in container_list:
        container_info = VL_MOD[container_name]
        Apptainer_file = _ContainerFull(container_info['Apptainer_file'])
        container_loc = CU.get_container_path(container_info)
        CU.print_container_info(container_name,Apptainer_file,container_loc)
        info.append([container_name,Apptainer_file,container_loc])
    print('This may take a while\n')

    for _info in info:
        CU.upgrade_container(*_info)


def handle_messages(client_socket, net_logger, parsed_args, gpu_flag, bind_points_default):

    while True:
        rec_dict = CU.receive_data(client_socket, parsed_args.debug)

        if rec_dict == None:
            CU.log_net_info(net_logger, "Socket has been closed")
            return
        event = rec_dict["msg"]

        container_id = rec_dict["Cont_id"]
        CU.log_net_info(
            net_logger,
            f'Server - received "{event}" event from container {container_id}',
        )

        pwd_dir = SU.get_pwd()
                 
        if event in ("Exec","MPI"):
            # will need to add option for docker when fixed
            
            cont_name = rec_dict["Cont_name"]
            cont_info = VL_MOD[cont_name]

            Apptainer_file = _ContainerFull(cont_info['Apptainer_file'])
            container_loc = CU.get_container_path(cont_info)
            CU.check_container(cont_name, Apptainer_file, container_loc)

            cont_info["container_path"] = Apptainer_file
            cont_info["container_cmd"] = f"apptainer exec --contain {gpu_flag} --writable-tmpfs -H {pwd_dir}"
            cont_info["bind"] = bind_points_default

            args = rec_dict.get("args", ())
            kwargs = rec_dict.get("kwargs", {})

            if event=='Exec':
                stdout = kwargs.get("stdout", None)
                if stdout is not None and not os.path.isdir(os.path.dirname(stdout)):
                    # stdout is a file path within VL_Manager so need to get the path on the host
                    stdout = CU.path_change_binder(stdout, bind_points_default)
                    kwargs["stdout"] = stdout
                    
                RC = CU.Exec_Container_Manager(cont_info, *args, **kwargs)
                CU.send_data(client_socket, RC, parsed_args.debug)
            else:
                # MPI
                # information needed when spawning a new process
                info = {'bind_points_default':bind_points_default,'gpu_flag':gpu_flag}
                shared_dir = rec_dict['shared_dir']
                with open(f"{shared_dir}/bind_points.pkl",'wb') as f:
                    pickle.dump(info,f)
                RC = CU.MPI_Container_Manager(cont_info, *args, **kwargs)
                CU.send_data(client_socket, RC, parsed_args.debug)    

def make_socket(host = '0.0.0.0', tcp_port=None):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setblocking(True)

    if tcp_port is None: 
        sock.bind((host, 0)) # will find a free port on the host
    else: 
        sock.bind((host, tcp_port)) # bind to tcp_port provided

    sock.listen(20)
    return sock

def process(vlab_dir, sock, *args):
    """Function that runs in a thread to handle communication ect."""
    cont_ready = threading.Event()
    sock_lock = threading.Lock()
    log_dir = f"{vlab_dir}/.log/network_log"
    net_logger = CU.setup_networking_log(log_dir)

    ################################
    while True:
        # check for new connections and them to list
        client_socket, client_address = sock.accept()

        # spawn a new thread to deal with messages
        thread = threading.Thread(
            target=handle_messages,
            args=(
                client_socket,
                net_logger,
                *args
            ),
        )
        thread.daemon = True
        thread.start()

def handle_messages2(client_socket, info):
    pwd_dir = SU.get_pwd()
    bind_points_default = info['bind_points_default']
    gpu_flag = info['gpu_flag']

    while True:
        rec_dict = CU.receive_data(client_socket)

        if rec_dict == None:
            return
        
        event = rec_dict["msg"]

        if event in ("Exec"):
            # will need to add option for docker when fixed
            
            container_cmd = f"apptainer exec --contain {gpu_flag} --writable-tmpfs -H {pwd_dir}"

            cont_name = rec_dict["Cont_name"]
            VL_MOD = SU.load_module_config(vlab_dir)
            cont_info = VL_MOD[cont_name]

            if cont_info["Apptainer_file"].startswith('/'):
                container_path = cont_info["Apptainer_file"] # full path provided
            else:
                container_path = f"{ContainerDir}/{cont_info['Apptainer_file']}" # relative path from container dir

            cont_info["container_path"] = container_path
            cont_info["container_cmd"] = container_cmd
            cont_info['bind'] = bind_points_default

            # check apptainer sif file exists and if not build from docker version
            if not os.path.exists(container_path):
                os.makedirs(os.path.dirname(container_path),exist_ok=True)
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


            stdout = kwargs.get("stdout", None)
            if stdout is not None and not os.path.isdir(os.path.dirname(stdout)):
                # stdout is a file path within VL_Manager so need to get the path on the host
                stdout = CU.path_change_binder(stdout, bind_points_default)
                kwargs["stdout"] = stdout
                
            RC = CU.Exec_Container_Manager(cont_info, *args, **kwargs)
            CU.send_data(client_socket, RC)

        elif event in ('kill'):
            client_socket.shutdown(socket.SHUT_RDWR)
            client_socket.close()
            return 1


            


def start_server(temp_file,shared_dir):
    # Create TCP port
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setblocking(True)
    sock.bind(('0.0.0.0', 0)) # will find a free port on the host
    sock.listen(20)
    # write TCP-port to temp_file so that bash script can use it
    with open(temp_file,'w') as f:
        f.write(str(sock.getsockname()[1]))
    # get bind information from file
    with open(f"{shared_dir}/bind_points.pkl",'rb') as f:
        info = pickle.load(f)
    while True:
        client_socket, client_address = sock.accept()
        a = handle_messages2(client_socket, info)
        if a==1:
            break

def kill_server(tcp_port):
    # create new socket
    host_name = socket.gethostname()
    sock = CU.create_tcp_socket(host_name,int(tcp_port))

    # Create info dictionary to send to VLserver. The msg 'Exec' calls Exec_Container_Manager
    # on the server, where  'args' and 'kwargs' are passed to it.
    info = {"msg": "kill"}

    # send data to relevant function in VLserver
    CU.send_data(sock, info)

def get_host(temp_file):
    ''' Used to get hostname when launching with MPI as environment variable not updated'''
    with open(temp_file,'w') as f:
        f.write(str(socket.gethostname()))




if __name__ == "__main__":
    main()