'''
        Script to enable comunication with and spawning of containers.
        #######################################################################
        Note: Current containers are:
        1 - Base VirtualLab
        2 - CIL
        3 - GVXR
        4 - Container tests
        #######################################################################
'''
import socket
import subprocess
import threading
import argparse
import os
import json
import sys
from pathlib import Path

#import Scripts.Common.VLContainer.VL_Modules as VL_MOD
import yaml
from Scripts.Common.VLContainer.Container_Utils import check_platform, \
    Format_Call_Str, send_data, receive_data, setup_networking_log,\
    log_net_info
def ContainerError(out,err):
    '''Custom function to format error message in a pretty way.'''
    Errmsg = "\n========= Container returned non-zero exit code =========\n\n"\
                f"std out: {out}\n\n"\
                f"std err:{err}\n\n"\
                "==================================\n\n"
    return Errmsg

def get_vlab_dir(parsed_dir):
    ''' 
    Function to get path to vlab_dir from either:
    input function parameters or os environment. in that order.
    If nether is possible it defaults to the users home directory.
    which will be either /home/{user}/VirtualLab 
    or C:\Documents\VirtualLab depending upon the OS.

    If the given directory does not exist it raises a value error.

    '''
    if parsed_dir:
       vlab_dir = Path(parsed_dir)
    else:
    # get dir from OS environment which should be set during installation
        vlab_dir = os.environ.get('VL_DIR',None)
        if vlab_dir == None:
            vlab_dir = Path.home() / 'VirtualLab'
        else:
            # here because you can't create a Path object from None
            vlab_dir = Path(vlab_dir)
        
    if not vlab_dir.is_dir():
        raise ValueError(f'Could not find VirtualLab install directory. The directory {str(vlab_dir)} does not appear to exist. \n' \
        ' Please specify where to find the VirtualLab install directory using the -d option.')

    return vlab_dir

def check_for_errors(process_list,client_socket,sock_lock):
    ''' 
    Function to take in nested a dictionary of containing running processes and container id's.
    Ideally any python errors will be handled and cleanup should print an error message to the screen
    and send a success message to the main process. Thus avoiding hanging the application.
    This function here is to catch any non-python errors. By simply running proc.communicate()
    to check each running process. From there if the return code is non zero it stops the server and
    spits out the std_err from the process.
     '''
    from socket import timeout
    from subprocess import TimeoutExpired
    if not process_list:
    # if list is empty return straight away and continue in while loop.
        return
    else:
        sock_lock.acquire()
        for proc in process_list.values():
            # check if the process has finished
            #communicate sets returncode inside proc if finished
            if proc.returncode is not None and proc.returncode != 0 :      
                #This converts the strings from bytes to utf-8 
                # however, we need to check they exist because
                # none objects can't be converted to utf-8
                if outs:
                    outs = str(outs,'utf-8')
                if errs:
                    errs = str(errs,'utf-8')
                
                err_mes = ContainerError(outs,errs)
                print(err_mes)
                #wait 5 seconds to see if error was caught in python
                # If so we should receive a finished message
                client_socket.settimeout(5)
                conn_timeout = False
                try:
                    message = receive_data(client_socket)
                except timeout:
                    conn_timeout = True
                if conn_timeout:
                    # error was either not python or was not caught in python
                    # send message to tell main vlab thread to close and 
                    # thus end the program.
                    data = {"msg":"Error","stderr":'-1'}
                    send_data(client_socket,data)
                elif message == 'Finished':
                    # Python has finished so error must have been handled there
                    # Thus no action needed from this end.
                    continue
                else:
                    ValueError("unexpected message {message} received on error.")
        sock_lock.release()
    return
#global variables for use in all threads
waiting_cnt_sockets = {}
target_ids = []
task_dict = {}
settings_dict = {}
running_processes = {}
next_cnt_id = 1
manager_socket = None

def load_module_config(vlab_dir):
    ''' Function to get the config for the 
    modules from VL_Modules.yaml file 
    '''
    #load module config from yaml_file
    config_file = vlab_dir / 'VL_Modules.yaml'
    with open(config_file)as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exception:
            print(exception)
    return config

def handle_messages(client_socket,net_logger,VL_MOD,sock_lock):
    global waiting_cnt_sockets
    global target_ids
    global task_dict
    global settings_dict
    global running_processes
    global next_cnt_id
    global manager_socket
    while True:
        rec_dict = receive_data(client_socket)
        if rec_dict == None:
            log_net_info(net_logger,'Socket has been closed')
            return
        event = rec_dict["msg"]
        container_id = rec_dict["Cont_id"]
        log_net_info(net_logger,f'Server - received "{event}" event from container {container_id}')
        if event == 'RunJob':
            Module = VL_MOD[rec_dict["Tool"]]
            num_containers = rec_dict["Num_Cont"]
            Cont_runs = rec_dict["Cont_runs"]
            param_master = rec_dict["Parameters_Master"]
            if rec_dict["Parameters_Var"] == 'None':
                param_var = None
            else:
                param_var = rec_dict["Parameters_Var"]
            project = rec_dict["Project"]
            simulation = rec_dict["Simulation"]
            # setup command to run docker or singularity
            if use_singularity:
                container_cmd = 'singularity exec --writable-tmpfs'
            else:
                # this monstrosity logs the user in as "themself" to allow safe access top x11 graphical apps"
                #see http://wiki.ros.org/docker/Tutorials/GUI for more details
     
                container_cmd = 'docker run '\
                                '--rm -it --network=host'\
                                '--env="DISPLAY" --env="QT_X11_NO_MITSHM=1" '\
                                '--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" '
            sock_lock.acquire()
                
                    
            # loop over containers once to create a dict of final container ids
            # and associated runs to output to file
            for Container in Cont_runs:
                target_ids.append(next_cnt_id)
                list_of_runs = Container[1]
                task_dict[str(next_cnt_id)] = list_of_runs
                settings_dict[str(next_cnt_id)]=rec_dict["Settings"]
                next_cnt_id += 1

            # loop over containers again to spawn them this time
            for n,Container in enumerate(Cont_runs):    
                options, command = Format_Call_Str(Module,vlab_dir,param_master,
                                param_var,project,simulation,use_singularity,target_ids[n])

                log_net_info(net_logger,f'Server - starting a new container with ID: {target_ids[n]} '
                                        f'as requested by container {container_id}')

                # spawn the container
                container_process = subprocess.Popen(f'{container_cmd} {options} {command}',
                    stderr = subprocess.PIPE, shell=True)
                    #add it to the list of waiting sockets
                waiting_cnt_sockets[str(target_ids[n])] = {"socket": client_socket, "id": str(target_ids[n])}

                running_processes[str(target_ids[n])] = container_process
                            
                    # send message to tell manager container what id the new containers will have
                data = {"msg":"Running","Cont_id":target_ids[n]}
                send_data(manager_socket, data)
            sock_lock.release()

        elif event == 'Ready':
            sock_lock.acquire()
            # containers are ready so send the list of tasks and id's to run
            data2 = {"msg":"Container_runs","tasks":task_dict[str(container_id)]
                    ,"settings":settings_dict[str(container_id)]}
            send_data(client_socket, data2)
            sock_lock.release()

        elif event == 'Finished':
            sock_lock.acquire()
            container_id = str(container_id)
            if container_id in waiting_cnt_sockets:
                log_net_info(net_logger,f'Server - container {container_id} finished working, '
                    f'notifying source container {waiting_cnt_sockets[container_id]["id"]}')
                waiting_cnt_socket = waiting_cnt_sockets[container_id]["socket"]
                data = {"msg":"Success","Cont_id":waiting_cnt_sockets[container_id]["id"],"target_id":int(container_id)}
                send_data(manager_socket,data)
                running_processes.pop(str(container_id))
            sock_lock.release()
            break
        else:
            client_socket.shutdown(socket.SHUT_RDWR)
            client_socket.close()
            raise ValueError(f'Unknown message {event} received')
        check_for_errors(running_processes,waiting_cnt_sockets["Manager"]["socket"],sock_lock)

def process(vlab_dir,use_singularity):
    ''' Function that runs in a thread to handle communication ect. '''
    global waiting_cnt_sockets
    global next_cnt_id
    global manager_socket
    net_logger = setup_networking_log()
    sock_lock = threading.Lock()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,1)
    sock.setblocking(True)
    host = "0.0.0.0"
    sock.bind((host, 9000))
    sock.listen(20)
    VL_MOD = load_module_config(vlab_dir)
    ###############
    # VLAB Started
    while True:
    # first while loop to wait for signal to start virtualLab
    # since the "VirtualLab started" message will only be sent 
    # by vl_manger we can use this to identify the socket for
    # the manger 
        manager_socket, manager_address = sock.accept()
        log_net_info(net_logger,f'received request for connection.')
        rec_dict = receive_data(manager_socket)
        event = rec_dict["msg"]
        if event == 'VirtualLab started':
            log_net_info(net_logger,f'received VirtualLab started')
            waiting_cnt_sockets["Manager"]={"socket": manager_socket, "id": 0}
            #spawn a new thread to deal with messages
            thread = threading.Thread(target=handle_messages,args=(manager_socket,net_logger,VL_MOD,sock_lock))
            thread.daemon = True 
            thread.start()
            break
        else:
        # we are not expecting any other message so raise an error 
        # as something has gone wrong with the timing.
            manager_socket.shutdown(socket.SHUT_RDWR)
            manager_socket.close()
            raise ValueError(f'Unknown message {event} received, expected VirtualLab started')
      
################################
    while True:
        #check for new connections and them to list
        client_socket, client_address = sock.accept()
        waiting_cnt_sockets[str(next_cnt_id)] = {"socket": client_socket, "id": next_cnt_id}
        next_cnt_id += 1
        #spawn a new thread to deal with messages
        thread = threading.Thread(target=handle_messages,args=(client_socket,net_logger,VL_MOD,sock_lock))
        thread.daemon = True 
        thread.start()

##########################################################################################
####################  ACTUAL CODE STARTS HERE !!!! #######################################
##########################################################################################
if __name__ == "__main__":
# rerad in CMD arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--vlab", help = "Path to Directory on host containing \
     VirtualLab (default is assumed to be curent working directory).", default=None)
    parser.add_argument("-f", "--Run_file", help = "Runfile to use (default is assumed to \
        be curent working directory).", default="Run.py")
    parser.add_argument("-D", "--Docker", help="Flag to use docker on Linux host instead of \
        defaulting to Singularity.This will be ignored on Mac/Windows as Docker is the default.",
        action='store_true')
    parser.add_argument("-C", "--Container", help="Flag to use tools in Containers.",
                        action='store_true')
    parser.add_argument("-T", "--test", help="Flag to initiate comms testing.",
                        action='store_true')

    args = parser.parse_args()
    # get vlab_dir either from cmd args or environment
    vlab_dir= get_vlab_dir(args.vlab)
# Set flag to allow cmd switch between singularity and docker when using linux host.
    use_singularity = check_platform() and not args.Docker
# set flag to run tests instate of the normal runfile
    if args.test:
        Run_file = f'Run_ComsTest.py'
    else:
        Run_file = args.Run_file

    Container = args.Container
    if Container:
        # start server listening for incoming jobs on separate thread
        lock = threading.Lock()
        thread = threading.Thread(target=process,args=(vlab_dir,use_singularity))
        thread.daemon = True

        Modules = load_module_config(vlab_dir)
        Manager = Modules["Manager"]
        thread.start()
        #start VirtualLab
        lock.acquire()
        if use_singularity:
            proc=subprocess.Popen(f'singularity exec --no-home --writable-tmpfs --nv -B \
                            /usr/share/glvnd -B {vlab_dir}:/home/ibsim/VirtualLab {Manager["Apptainer_file"]} '
                            f'{Manager["Run_script"]} -f /home/ibsim/VirtualLab/RunFiles/{Run_file}', shell=True)
        else:
            # Assume using Docker
            proc=subprocess.Popen(f'docker run --rm -it --network=host -v {vlab_dir}:/home/ibsim/VirtualLab ' f'{Manager["Docker_url"]}:{Manager["Tag"]} ' \
                            f'"{Manager["Run_script"]} -f /home/ibsim/VirtualLab/RunFiles/{Run_file}"', shell=True)
        lock.release()
        # wait until virtualLab is done before closing
        proc.wait()
    else:
        # use native version
        print("Warning: VirtualLab is Running in native mode Some tools will be unavailable.\n"
            " Since version 2.0 VirtualLab is being setup to run inside a container with \n "
            " Docker/Singularity. To use this container functionality, and tools which depend \n"
            " on it you will need to install docker or Singularity and pass in the -C option. \n")

        proc=subprocess.check_call(f'VL_Manager -f {Run_file}', shell=True)





