from ast import Raise
import socket
import sys
import subprocess
import threading
import argparse
import os
import json
''' Script to enable comunication with and spawning of containers.
        #######################################################################
        Note: Current container ID's are:
        1 - Base VirtualLab
        2 - CIL

        If/when you want to add more containers you will need to give 
        them a unique id in the container_id's dict in VL_sever.py 
        and add it to this list as a courtesy so everyone is on the same page
        #######################################################################
'''
waiting_cnt_sockets = {}

Container_IDs = {"VLab":1,"CIL":2}
def Format_Call_Str(Tool,vlab_dir,param_master,param_var,Project,Simulation):
    ''' Function to format string for bindpoints and container to call specified tool.'''
##### Format cmd argumants #########
    if param_var is None:
        param_var = ''
    else:
        param_var = '-v ' + param_var
    
    param_master = '-m '+ param_master
    Simulation = '-s ' + Simulation
    Project = '-p ' + Project
#########################################
# Format run string and script to run   #
# container based on tool used.         #
#########################################
    if Tool == "CIL":
        call_string = '-B /run:/run -B .:/home/ibsim/VirtualLab CIL_sand'
        command = '/home/ibsim/VirtualLab/Run_CIL.sh {} {} {} {}'.format(param_master,param_var,Project,Simulation)
        print(command)

    elif Tool == "GVXR":
        call_string = '-B {}:/home/ibsim/VirtualLab VL_GVXR.sif'.format(vlab_dir)
        command = 'python'
        
    # Add others as need arises
    else:
        Raise(ValueError("Tool not recognised as callable in container."))
    return call_string, command

def process(vlab_dir):
    lock = threading.Lock()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 9999))
    sock.listen(20)

    while True:
        client_socket, client_address = sock.accept()
        data = client_socket.recv(1024).decode('utf-8')
        #event, container_id, Tool = data.split(':')
        rec_dict = json.loads(data)
        event = rec_dict["msg"]
        container_id = rec_dict["Cont_id"]
        print('Server - received "{}" event from container {} at {}'.format(event, container_id, client_address))

        if event == 'VirtualLab started':
            client_socket.close()
        elif event == 'RunJob':
            Tool = rec_dict["Tool"]
            param_master = rec_dict["Parameters_Master"]
            param_var = rec_dict["Parameters_Var"]
            Project = rec_dict["Project"]
            Simulation = rec_dict["Simulation"]

            lock.acquire()
            call_string, command = Format_Call_Str(Tool,vlab_dir,param_master,param_var,Project,Simulation)
            target_id = Container_IDs[Tool]
            print('Server - starting a new container with ID: {} '
                  'as requested by container {}'.format(target_id, container_id))
            try:
                proc = subprocess.check_call('singularity exec --contain --writable-tmpfs --nv {} {}'.format(call_string,command), shell=True)
            except Exception:
                lock.release()
                client_socket.close()
                raise 
            waiting_cnt_sockets[str(target_id)] = {"socket": client_socket, "id": container_id}
            lock.release()
            client_socket.close()

        elif event == 'finished':
            lock.acquire()
            container_id = str(container_id)
            if container_id in waiting_cnt_sockets:
                print('Server - container {} finished working, '
                      'notifying source container {}'.format(container_id, waiting_cnt_sockets[container_id]["id"]))
                waiting_cnt_socket = waiting_cnt_sockets[container_id]["socket"]
                waiting_cnt_socket.sendall('Success'.encode())
                waiting_cnt_socket.close()
            lock.release()
            client_socket.close()
        else:
            client_socket.close()
            raise ValueError()

if __name__ == "__main__":
# rerad in CMD arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--vlab", help = "Path to Directory ion host containing VirtualLab (default is assumed to be curent working directory).", default=os.getcwd())
    parser.add_argument("-i", "--Run_file", help = "Runfile to use (default is assumed to be curent working directory).", default="Run.py")
    args = parser.parse_args()
    vlab_dir=os.path.abspath(args.vlab)
    Run_file = args.Run_file
    # start server listening for incoming jobs on seperate thread
    lock = threading.Lock()
    thread = threading.Thread(target=process,args=(vlab_dir,))
    thread.daemon = True

    thread.start()
    #start VirtualLab
    lock.acquire()
    proc=subprocess.Popen('singularity exec --no-home --writable-tmpfs --nv -B /usr/share/glvnd -B {}:/home/ibsim/VirtualLab VL_GVXR.sif '
                    'VirtualLab -f /home/ibsim/VirtualLab/RunFiles/{}'.format(vlab_dir,Run_file), shell=True)
    
    lock.release()
    # wait untill virtualLab is done before closing
    proc.wait()
