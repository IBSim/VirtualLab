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
import time

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

def check_for_errors(process_list,client_socket,sock_lock,debug):
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
        for proc in process_list.values():
            proc.poll()
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
                    message = receive_data(client_socket,debug)
                except timeout:
                    conn_timeout = True
                if conn_timeout:
                    # error was either not python or was not caught in python
                    # send message to tell main vlab thread to close and 
                    # thus end the program.
                    data = {"msg":"Error","stderr":'-1'}
                    send_data(client_socket,data,debug)
                elif message == 'Finished':
                    # Python has finished so error must have been handled there
                    # Thus no action needed from this end.
                    continue
                else:
                    ValueError("unexpected message {message} received on error.")
    return
#global variables for use in all threads
waiting_cnt_sockets = {}
target_ids = []
task_dict = {}
settings_dict = {}
running_processes = {}
run_arg_dict = {}
tool_dict = {}
next_cnt_id = 1
manager_socket = None
cont_ready = False

def load_module_config_yaml(vlab_dir):
    ''' Function to get the config for the 
    modules from VL_Modules.yaml file 
    '''
    #load module config from yaml_file
    config_file = vlab_dir / 'Config/VL_Modules.yaml'
    with open(config_file)as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exception:
            print(exception)
    return config

def load_module_config(vlab_dir):
    ''' Function to get the config from a json file 
    '''
    import json
    from pathlib import Path
    vlab_dir = Path(vlab_dir)
    #load module config from file
    config_file = vlab_dir/'Config/VL_Modules.json'
    with open(config_file)as file:
        config = json.load(file)
    return config

def handle_messages(client_socket,net_logger,VL_MOD,sock_lock,cont_ready,debug):
    global waiting_cnt_sockets
    global target_ids
    global task_dict
    global settings_dict
    global running_processes
    global next_cnt_id
    global manager_socket
    global run_arg_dict
    global tool_dict
    # list of messages to simply relay from Container_id to Target_id
    relay_list = ["Continue","Waiting","Error"]
    while True:
        rec_dict = receive_data(client_socket,debug)
        if rec_dict == None:
            log_net_info(net_logger,'Socket has been closed')
            return
        event = rec_dict["msg"]
        container_id = rec_dict["Cont_id"]
        log_net_info(net_logger,f'Server - received "{event}" event from container {container_id}')
        if event == 'Spawn_Container':
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
            # setup command to run docker or Apptainer
            if use_Apptainer:
                container_cmd = 'Apptainer exec --writable-tmpfs'
            else:
                # this monstrosity logs the user in as "themself" to allow safe access top x11 graphical apps"
                #see http://wiki.ros.org/docker/Tutorials/GUI for more details
     
                container_cmd = 'docker run '\
                                '--rm -it --network=host'\
                                '--env="DISPLAY" --env="QT_X11_NO_MITSHM=1" '\
                                '--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" '
                 
            # loop over containers once to create a dict of final container ids
            # and associated runs to output to file
            sock_lock.acquire()
            target_ids = []
            for Container in Cont_runs:
                target_ids.append(next_cnt_id)
                list_of_runs = Container[1]
                task_dict[str(next_cnt_id)] = list_of_runs
                settings_dict[str(next_cnt_id)]=rec_dict["Settings"]
                run_arg_dict[str(next_cnt_id)] = rec_dict["run_args"]
                tool_dict[str(next_cnt_id)] = rec_dict["Tool"]
                next_cnt_id += 1

            # loop over containers again to spawn them this time
            for n,Container in enumerate(Cont_runs):    
                options, command = Format_Call_Str(Module,vlab_dir,param_master,
                                param_var,project,simulation,use_Apptainer,target_ids[n])

                log_net_info(net_logger,f'Server - starting a new container with ID: {target_ids[n]} '
                                        f'as requested by container {container_id}')

                # spawn the container
                container_process = subprocess.Popen(f'{container_cmd} {options} {command}',
                    shell=True)
                    #add it to the list of waiting sockets
                waiting_cnt_sockets[str(target_ids[n])] = {"socket": client_socket, "id": str(target_ids[n])}

                running_processes[str(target_ids[n])] = container_process
                            
                    # send message to tell manager container what id the new containers will have
                data = {"msg":"Running","Cont_id":target_ids[n]}
                send_data(client_socket, data,debug)
            sock_lock.release()
            # cont_ready should be set by another thread when the container messages to say its ready to go.
            # This loop essentially checks to see if the container started correctly by waiting for 10 seconds
            #  and if cont_ready is not set it will raise an error.
            ready = cont_ready.wait(timeout=15)
            # we've heard nothing from the container so we have 
            # to assume it has hung. Thus send error to manger
            # and client (if client is not the manger) 
            # to kill process. 
            # Note: to this should also kill the server.
            if not ready:
                data = {"msg":"Error","stderr":'-1'}
                if client_socket != manager_socket:
                    send_data(client_socket,data,debug)
                send_data(manager_socket,data)
                raise TimeoutError('The container appears to have have not started correctly.')

        elif event == 'Ready':
            cont_ready.set()
            log_net_info(net_logger,f'Server - Received message to say container {container_id} '
                            f'is ready to start runs.')
            # containers are ready so send the list of tasks and id's to run
            sock_lock.acquire()
            data2 = {"msg":"Container_runs","tasks":task_dict[str(container_id)]
                    ,"settings":settings_dict[str(container_id)],
                    "run_args":run_arg_dict[str(container_id)],
                    "Tool":tool_dict[str(container_id)]}
            sock_lock.release()       
            send_data(client_socket, data2,debug)
            # This function will run until the server receives "finished"
            #  or an error occurs in the container.
            check_pulse(client_socket,sock_lock,net_logger,debug)
            #client_socket.shutdown(socket.SHUT_RDWR)
            #client_socket.close()
            break
        elif event in relay_list:
            Target_id = str(rec_dict['Target_id'])
            Target_socket = waiting_cnt_sockets[Target_id]
            send_data(Target_socket, rec_dict,debug)
        else:
            client_socket.shutdown(socket.SHUT_RDWR)
            client_socket.close()
            raise ValueError(f'Unknown message {event} received')
    
def check_pulse(client_socket,sock_lock,net_logger,debug):
    ''' 
    Function to check for periodic messages from the containers to say 
    they are still running.
    '''
    global waiting_cnt_sockets
    global target_ids
    global task_dict
    global settings_dict
    global running_processes
    global next_cnt_id
    global manager_socket
    from socket import timeout
# wait up to 30 seconds to see if container has started or has heartbeat
# If not raise an error.
    client_socket.settimeout(30)
                
    while True:
        # check_for_errors(running_processes, client_socket, sock_lock)
        try:
            rec_dict = receive_data(client_socket,debug)
        except timeout:
            # we've heard nothing from the container so we have 
            # to assume it has hung. Thus send error to manger 
            # to kill process. Note this should also kill the server.
            data = {"msg":"Error","stderr":'-1'}
            send_data(client_socket,data,debug)
            raise TimeoutError('The container appears to have has hung')
        
        if rec_dict == None:
            log_net_info(net_logger,'Socket has been closed')
            return
        event = rec_dict["msg"]
        container_id = rec_dict["Cont_id"]
        if event == 'Finished':
            # Container is done so cleanup
            sock_lock.acquire()
            container_id = str(container_id)
            if container_id in waiting_cnt_sockets:
                log_net_info(net_logger,f'Server - container {container_id} finished working, '
                    f'notifying source container {waiting_cnt_sockets[container_id]["id"]}')
                waiting_cnt_socket = waiting_cnt_sockets[container_id]["socket"]
                data = {"msg":"Success","Cont_id":waiting_cnt_sockets[container_id]["id"],"Target_id":int(container_id)}
                send_data(manager_socket,data,debug)
                running_processes.pop(str(container_id))
            sock_lock.release()
            return
    
        elif event == "Beat":
            # got heartbeat to say container is still running
            log_net_info(net_logger,f'Server - Got heartbeat message from container {container_id}.')
        else:
            client_socket.shutdown(socket.SHUT_RDWR)
            client_socket.close()
            raise ValueError(f'Unexpected message {event} received from container {container_id}')
        
def process(vlab_dir,use_Apptainer,debug):
    ''' Function that runs in a thread to handle communication ect. '''
    global waiting_cnt_sockets
    next_cnt_id = 1
    global manager_socket
    cont_ready = threading.Event()
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
        rec_dict = receive_data(manager_socket,debug)
        event = rec_dict["msg"]
        if event == 'VirtualLab started':
            log_net_info(net_logger,f'received VirtualLab started')
            waiting_cnt_sockets["Manager"]={"socket": manager_socket, "id": 0}
            #spawn a new thread to deal with messages
            thread = threading.Thread(target=handle_messages,args=(manager_socket,net_logger,VL_MOD,sock_lock,cont_ready,debug))
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
        thread = threading.Thread(target=handle_messages,args=(client_socket,net_logger,VL_MOD,sock_lock,cont_ready,debug))
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
        be current working directory).", default="Run.py")
    parser.add_argument("-D", "--Docker", help="Flag to use docker on Linux host instead of \
        defaulting to Apptainer.This will be ignored on Mac/Windows as Docker is the default.",
        action='store_true')
    parser.add_argument("-U", "--dry-run", help="Flag to update containers without running.",
                        action='store_true')
    parser.add_argument("-X", "--debug", help="Flag to print debug messages for networking.",
                        action='store_true')                        
    parser.add_argument("-T", "--test", help="Flag to initiate comms testing.",
                        action='store_true')

    args = parser.parse_args()
    # get vlab_dir either from cmd args or environment
    vlab_dir= get_vlab_dir(args.vlab)
# Set flag to allow cmd switch between Apptainer and docker when using linux host.
    use_Apptainer = check_platform() and not args.Docker
# set flag to run tests instate of the normal runfile
    if args.test:
        Run_file = f'Run_ComsTest.py'
    else:
        Run_file = args.Run_file
    path = vlab_dir / Path('RunFiles/') / Run_file
    if not path.exists():
        raise ValueError(f'Runfile not found. This must be a file inside the directory {vlab_dir}/RunFiles.')

    # start server listening for incoming jobs on separate thread
    lock = threading.Lock()
    thread = threading.Thread(target=process,args=(vlab_dir,use_Apptainer,args.debug))
    thread.daemon = True

    Modules = load_module_config(vlab_dir)
    Manager = Modules["Manager"]
    thread.start()
    #start VirtualLab
    lock.acquire()
    if use_Apptainer:
        proc=subprocess.Popen(f'Apptainer exec --no-home --writable-tmpfs --nv -B \
                        /usr/share/glvnd -B {vlab_dir}:/home/ibsim/VirtualLab {Manager["Apptainer_file"]} '
                        f'{Manager["Startup_cmd"]} -f /home/ibsim/VirtualLab/RunFiles/{Run_file}', shell=True)
    else:
        # Assume using Docker
        proc=subprocess.Popen(f'docker run --rm -it --network=host -v {vlab_dir}:/home/ibsim/VirtualLab ' f'{Manager["Docker_url"]}:{Manager["Tag"]} ' \
                            f'"{Manager["Startup_cmd"]} -f /home/ibsim/VirtualLab/RunFiles/{Run_file}"', shell=True)
    lock.release()
    # wait until virtualLab is done before closing
    proc.wait()






