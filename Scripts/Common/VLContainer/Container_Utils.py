from email import message
import socket
import json
import pickle
from types import SimpleNamespace as Namespace
from ast import Raise
import struct

def RunJob(Cont_id,Tool,Num_Cont,Cont_runs,Parameters_Master,Parameters_Var,Project,Simulation):
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
        Cont_id (int): unique Id of the container that is calling the function.
                For Now this should be set to 1 for Vlab. This has been
                deliberately left in to allow other containers to call 
                tools should the need arise.

        Tool (str): String containing name of the tool you want to run.

        Parameters_Master/Var (str): path INSIDE THE CONTAINER to the parameters file
        to be read in. Note you will need to setup directory binding carefully in VLsever.py.

        Cont_runs:  This is a list of tuples that maps runs to specific containers. 
        The first index is a container number and the second is a list
         of runs to be processed within said container.
        ''' 
    if Parameters_Var == None:
        Parameters_Var = 'None'
        # setup networking to communicate with host script whilst running in a container
    data = {"msg":"RunJob","Cont_id":Cont_id,"Tool":Tool,"Num_Cont":Num_Cont,"Cont_runs":Cont_runs,
            "Parameters_Master":Parameters_Master,"Parameters_Var":Parameters_Var,
            "Project":Project,"Simulation":Simulation}
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
    if isinstance(Parameters_Master,Namespace):
        raise Exception("Passing in Namespaces is not currently supported for Container tools. \
        These must be strings pointing to a Runfile.")
    
    #data_string = json.dumps(data)
    sock = create_tcp_socket()
    # send a signal to VL_server saying you want to run a CIL container
    send_data(sock, data)
    target_ids = []
    #wait to recive message saying the tool is finished before continuing on.
    while True:
        rec_dict=receive_data(sock)
        if rec_dict:
            if rec_dict['msg'] == 'Running':
                target_ids.append(rec_dict['Cont_id'])
        # wait until all containers have started
            if len(target_ids) == Num_Cont:      
                break
    
    while True:
        rec_dict = receive_data(sock)
        if rec_dict:
            if rec_dict['msg'] == 'Success' and rec_dict['Cont_id'] == 1:
                target_ids.remove(rec_dict['target_id'])
            if len(target_ids) == 0:
                container_return = '0'
                break
            if rec_dict['msg'] == 'Error':
                container_return = '-1'
                break
            #continue
    sock.close()
    return container_return

def create_tcp_socket():
    ''' function to create the tcp socket and connect to it.'''
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,1)
    sock.connect(("0.0.0.0", 9999))
    return sock

def Cont_Started(Cont_id):
    ''' Function to send a Message to the main script to say the container has started.'''
    data = {"msg":"started","Cont_id":Cont_id}
    sock = create_tcp_socket()
    send_data(sock, data)
    sock.close()
    return

def Cont_Finished(Cont_id):
    ''' Function to send a Message to the main script to say the container has Finished.'''
    data = {"msg":"Finished","Cont_id":Cont_id}
    sock = create_tcp_socket()
    send_data(sock, data)
    sock.close()
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
    
def receive_data(conn,payload_size=2048,debug=False):
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

def Format_Call_Str(Tool,vlab_dir,param_master,param_var,Project,Simulation,use_singularity,cont_id):
    ''' Function to format string for bind points and container to call specified tool.'''
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
# container based on tool used.         #
#########################################
# Setup command to run inside container and bind directories based on tool used    
    if Tool == "CIL":
        if use_singularity:
            call_string = f'-B /run:/run -B {vlab_dir}:/home/ibsim/VirtualLab \
                            --nv Containers/CIL_sand'
        else:
            call_string = f'-v /run:/run -v {vlab_dir}:/home/ibsim/VirtualLab --gpus all ibsim/vl_cil'

        command = f'/home/ibsim/VirtualLab/Containers/Run_CIL.sh \
                   {param_master} {param_var} {Project} {Simulation} {ID}'

    elif Tool == "GVXR":
        if use_singularity:
            call_string = f'-B /run:/run -B /dev/dri:/dev/dri -B {vlab_dir}:/home/ibsim/VirtualLab --nv Containers/GVXR_test.sif'
        else:
            call_string = f'-v /run:/run -v /dev:/dev -v {vlab_dir}:/home/ibsim/VirtualLab -e QT_X11_NO_MITSHM=1 --gpus all ibsim/vl_gvxr'

        command = f'/home/ibsim/VirtualLab/Containers/Run_GVXR.sh \
                   {param_master} {param_var} {Project} {Simulation} {ID}'
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