from email import message
import socket
import json
import pickle
from types import SimpleNamespace as Namespace
from ast import Raise
import struct

def Spawn_Container(VL,**kwargs):
    ''' Function to enable communication with host script from container.
        This will be called from the VirtualLab container to Run a job 
        with another toll in separate container. At the moment this 
        is only CIL but other tools can/will be added in time.
        #######################################################################
        Note: Current container ID's are:
        1 - Base VirtualLab
        2 - CIL

        If/when you want to add more containers you will need to give 
        them a unique id in the container_id's dict in VL_sever.py 
        and add it to this list as a courtesy so everyone is on the same page
        #######################################################################
        Inputs: 
        a dict containing the following parameters to setup the job:

        Cont_id: ID of the container requesting the job. This is usually but not
                 necessarily VL_manager (i.e. container 1).
        Tool: string to identify the module to spin up.
        Num_Cont: maximum number of containers to run in parallel,
        Parameters_Master: string pointing to appropriate Runfile
        Parameters_Var: string pointing to appropriate Runfile or None
        Project: Name of the project.
        Simulation: Name of the Simulation
        Settings: dict of settings that were passed into VL_Manger. 
                  This ensures modules receive the same settings 
                  (mode,Nbjobs ect.)
        #######################################################
        ####################   Note:   ########################
        #######################################################
         The return value for this function will be different 
         depending upon if it is called by VLSetup or VLModule.
         see comment on line 81 for more details.
        ''' 
    waiting_containers = {}

    if kwargs['Parameters_Var'] == None:
        kwargs['Parameters_Var'] = 'None'

    kwargs['msg'] = "Spawn_Container"
    # Long Note: we are fully expecting Parameters_Master and Parameters_Var to be strings 
    # pointing to Runfiles. However base VirtualLab supports passing in Namespaces.
    # (see virtualLab.py line 178 and GetParams for context). 
    # For the sake of my sanity we assume you are using normal RunFiles, which most users 
    # likely are.
    #
    # Therefore this check is here to catch any determined soul and let you know if you 
    # want to pass in Namespaces for container tools you will need implement it yourself. 
    # In principle this means converting Namespace to a dict with vars(Parameters_Master).
    # Then sending it over as json string and converting it back on the other side.
    # In practice you may run into issues with buffer sizes for sock.recv as the strings
    # can get very long indeed.
    if isinstance( kwargs['Parameters_Master'],Namespace):
        raise Exception("Passing in Namespaces is not currently supported for Container tools. \
        These must be strings pointing to a Runfile.")
    sock = kwargs['tcp_socket']
    kwargs.pop('tcp_socket')
    vltype = kwargs['Method_Name']
    #data_string = json.dumps(data)
    # send a signal to VL_server saying you want to spawn a container
    send_data(sock, kwargs,VL.debug)
    target_ids = []
    #wait to receive message saying the tool is finished before continuing on.
    while True:
        rec_dict=receive_data(sock,VL.debug)
        if rec_dict:
            if rec_dict['msg'] == 'Running':
                target_ids.append(rec_dict['Cont_id'])
        # wait until all containers have started
            if len(target_ids) == kwargs['Num_Cont']:      
                break
    # check if container spawned by module or manager. If called by manger (VLsetup)
    # we should wait for the containers to complete. VLModule on the other hand is 
    # expecting a list of containers and it will handle the rest with a combination of
    #  calls to: Wait_For_Container, Cont_continue and Cont_Waiting.
    if VL.__class__.__name__ == 'VLModule':
        return target_ids

    while True:
        rec_dict = receive_data(sock,VL.debug)
        if rec_dict:
            if rec_dict['msg'] == 'Success':
                target_ids.remove(rec_dict['Target_id'])
            elif rec_dict['msg'] == 'Error':
                container_return = '-1'
                break
            else: 
                continue

        if len(target_ids) == 0:
            container_return = '0'
            break
    #end of while loop
    if container_return == '0':
        VL.do_Analytics(vltype)
    return container_return

def create_tcp_socket(port=9000):
    ''' Function to create the tcp socket and connect to it. 
        The default port is 9000 for coms with the containers. 
        This however should be set to 5000 for coms 
        with the host process. '''
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,1)
    sock.setblocking(True)
    host = "0.0.0.0"
    sock.connect((host, port))   
    return sock

def Cont_Started(Cont_id,sock,debug=False):
    ''' Function to send a Message to the main script to say the container has started.'''
    data = {"msg":"started","Cont_id":Cont_id}
    send_data(sock, data)
    sock.close()
    return

def Cont_Finished(Cont_id,sock,debug=False):
    ''' Function to send a Message to the main script to say the container has Finished.'''
    data = {"msg":"Finished","Cont_id":Cont_id}
    send_data(sock, data,debug)
    sock.close()
    return

def Wait_For_Container(sock,Cont_id,Target_id,debug=False):
    '''
    Function to wait to receive a message from container Target_id to say if 
    it has finished or is waiting. 
    The return value can be used to determine if the target container. 
    Completed (successfully or not) or is simply waiting for the signal
    to continue.
    '''
    while True:
        rec_data = receive_data(sock,debug)
        if rec_data == None:
            import sys
            sys.exit(f'got unexpected socket shutdown whilst waiting for container {Target_id}.')
        #check if the message is for us
        if rec_dict["Target_id"] != Cont_id:
            continue
        elif rec_dict["msg"] == "Waiting":
            return "Waiting"
        elif rec_dict["msg"] == "Finished":
            return "Finished"
        elif rec_dict["msg"] == "Error":
            return "Error"
        else:
            sock.shutdown(socket.SHUT_RDWR)
            sock.close()
            raise ValueError(f'Unknown message {rec_dict["msg"]} received')
        return

def Cont_Continue(Cont_id,sock,Target_id,wait=True,debug=False):
    ''' 
    Function to send a Message to a waiting container (Target_id) to tell it to continue working.
    optional arguments wait and Finished are flags to say if you wish to wait for 
    the container.
    '''
    data = {"msg":"Continue","Cont_id":Cont_id}
    send_data(sock, data,debug)
    status = ''
    if wait:
        status = Wait_For_Container()
    return

def Cont_Waiting(Cont_id,target_id, sock,debug=False):
    ''' 
    Function to send a Message to container Target_id say the current 
    container is waiting for a message to continue.
    '''
    data = {"msg":"Waiting","Cont_id":Cont_id,"Target_id":Target_id}
    send_data(sock, data,debug)
    # wait to receive message to continue
    rec_data = receive_data(sock,debug)
    if rec_data == None:
        import sys
        sys.exit(f'Waiting container {Cont_id} got unexpected socket shutdown')
    elif rec_dict["msg"] == "Continue":
        return
    else:
        sock.shutdown(socket.SHUT_RDWR)
        sock.close()
        raise ValueError(f'Unknown message {rec_dict["msg"]} received')
    return


def send_data(conn, payload,bigPayload=False,debug=False):
    '''
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
    '''
    # serialize payload
    if debug:
        print(f'sent:{payload}')
    serialized_payload = json.dumps(payload).encode('utf-8')
    payload_size = len(serialized_payload)
    if  payload_size > 2048 and not bigPayload:
        print("###################################################\n"\
            f"Warning: Payload has a size of {payload_size} bytes.\n"\
        "This exceeds the standard buffer size of 2048 bytes.\n"\
        "You will need to ensure you set the buffer on the \n"\
        "corresponding call to receive_data to a large \n"\
        "enough value or else data may be lost/corrupted.\n"\
        "To suppress this message set the bigPayload flag.\n"\
        "###################################################")
    conn.sendall(serialized_payload)
    
def receive_data(conn,debug,payload_size=2048):
    '''
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

    '''
    received_payload = conn.recv(payload_size)

    if not received_payload:
        payload = None
    else:
        received_payload=received_payload.decode('utf-8')
        payload = json.loads(received_payload)
        if debug:
            print(f'received:{payload}')
    return (payload)

def Format_Call_Str(Module,vlab_dir,param_master,param_var,Project,Simulation,use_Apptainer,cont_id):
    ''' Function to format string for bind points and container to call specified tool.'''
    import os
    import subprocess
##### Format cmd argumants #########
    if param_var is None:
        param_var = ''
    else:
        param_var = '-v ' + param_var

    param_master = '-m '+ param_master
    Simulation = '-s ' + Simulation
    Project = '-p ' + Project
    ID = '-I '+ str(cont_id)
#########################################
# Format run string and script to run   #
# container based on Module used.       #
#########################################
    if use_Apptainer:
        update_container(Module,vlab_dir)
        call_string = f' -B /run:/run -B /tmp:/tmp -B {str(vlab_dir)}:/home/ibsim/VirtualLab \
                        {str(vlab_dir)}/{Module["Apptainer_file"]}'
    else:
        #docker
        call_string = f'-v /run:/run -v /tmp:/tmp -v {str(vlab_dir)}:/home/ibsim/VirtualLab {Module["Docker_url"]}:{Module["Tag"]} '
    
    # get custom command line arguments if specified in config.
    arguments = Module.get("cmd_args",None)
    if arguments == None:
        command = f'{Module["Startup_cmd"]} \
               {param_master} {param_var} {Project} {Simulation} {ID}'
    else:
        command = f'{Module["Startup_cmd"]} {arguments}'

    return call_string, command

def check_platform():
    '''Simple function to return True on Linux and false on Mac/Windows to
    allow the use of Apptainer instead of Docker on Linux systems.
    Apptainer does not support Windows/Mac OS hence we need to check.
    Note: Docker can be used on Linux with the --docker flag. This flag
    however is ignored on both windows and Mac since they already
    default to Docker.'''
    import platform
    use_Apptainer=False
    if platform.system()=='Linux':
        use_Apptainer=True
    return use_Apptainer

def setup_networking_log(filename):
    ''' 
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
    '''
    import logging
    from logging.handlers import TimedRotatingFileHandler
    import datetime
    now = datetime.datetime.now()
    today = now.strftime('%Y-%m-%d')
    filename=f'{filename}_{today}.log'
    log = logging.getLogger('logger')
    # Sets the base level for all logging.
    # Setting this to debug ensures we log everything.
    # Since default level is Warning if we didn't
    # set this and used debug in one of our handlers
    # it wouldn't log anything below warning. 
    log.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(message)s')

    # Logger for file
    fh = logging.FileHandler(filename, mode='a', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)

    # Logger for screen
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    log.addHandler(ch)

    # print date and time to log for starting virtualLab
    
    log.debug(f'started VirtualLab:{now}')
    return log

def log_net_info(logger,message,screen=False):
    if screen:
        logger.info(message)
    else:
        logger.debug(message)

def update_container(Module,vlab_dir):
    import os
    import subprocess
    Apptainer_file = f"{vlab_dir}/{Module['Apptainer_file']}"
    # check apptainer sif file exists and if not build from docker version
    if not os.path.exists(Apptainer_file):
        print(f"Apptainer file {Apptainer_file} does not appear to exist so building. This may take a while.")
        try:
            proc=subprocess.check_call(f'apptainer build '\
               f'{Apptainer_file} docker://{Module["Docker_url"]}:{Module["Tag"]}', shell=True)
        except subprocess.CalledProcessError as E:
            print(E.stderr)
            raise E
    return

def get_vlab_dir(parsed_dir=None):
    ''' 
    Function to get path to vlab_dir from either:
    input function parameters or os environment. in that order.
    If nether is possible it defaults to the users home directory.
    which will be either /home/{user}/VirtualLab 
    or C:\Documents\VirtualLab depending upon the OS.

    If the given directory does not exist it raises a value error.

    '''
    import os
    from pathlib import Path
    if parsed_dir != None:
       vlab_dir = Path(parsed_dir)
       os.environ['VL_DIR'] = str(parsed_dir)
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

def host_to_container_path(filepath):
    '''
    Function to Convert a path in the virtualLab directory on the host 
    to an equivalent path inside the container. since the vlab _dir is 
    mounted as /home/ibsim/VirtualLab inside the container.
    Note: The filepath needs to be absolute and  is converted
    into a string before it is returned.
    '''
    vlab_dir=get_vlab_dir()
    #location of vlab inside the container
    cont_vlab_dir = "/home/ibsim/VirtualLab"
    # convert path to be relative to container not host
    filepath = str(filepath).replace(str(vlab_dir), cont_vlab_dir)
    return filepath

def container_to_host_path(filepath):
    '''
    Function to Convert a path inside the container 
    to an equivalent path on the host. since the vlab _dir is 
    mounted as /home/ibsim/VirtualLab inside the container.

    Note: The filepath needs to be absolute and  is converted
    into a string before it is returned. 
    '''
    vlab_dir=get_vlab_dir()
    #location of vlab inside the container
    cont_vlab_dir = "/home/ibsim/VirtualLab"
    # convert path to be relative to host not container
    filepath = str(filepath).replace(cont_vlab_dir,str(vlab_dir))
    return filepath