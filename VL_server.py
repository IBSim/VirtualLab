from ast import Raise
import socket
import sys
import subprocess
import threading
import argparse
import os
import json
from Scripts.Common.VLContainer.container_tools import check_platform,Format_Call_Str
''' Script to enable comunication with and spawning of containers.
        #######################################################################
        Note: Current container ID's are:
        1 - Base VirtualLab
        2 - CIL
        3 - GVXR

        If/when you want to add more containers you will need to give 
        them a unique id in the container_id's dict in VL_sever.py 
        and add it to this list as a courtesy so everyone is on the same page
        #######################################################################
'''
waiting_cnt_sockets = {}

Container_IDs = {"VLab":1,"CIL":2,"GVXR":3}

def process(vlab_dir,use_singularity):
    lock = threading.Lock()
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,1)
    sock.bind(("0.0.0.0", 9999))
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
            options, command = Format_Call_Str(Tool,vlab_dir,param_master,param_var,Project,Simulation,use_singularity)
            target_id = Container_IDs[Tool]
            print('Server - starting a new container with ID: {} '
                  'as requested by container {}'.format(target_id, container_id))
            # setup comand to run docker or singularity
            if use_singularity:
                container_cmd = 'singularity exec --contain --writable-tmpfs'
            else:
                container_cmd = 'docker run -it'

            try:
                proc = subprocess.check_call('{} {} {}'.format(container_cmd,options,command), shell=True)
            except Exception:
                lock.release()
                client_socket.shutdown(socket.SHUT_RDWR)
                client_socket.close()
                raise 
            waiting_cnt_sockets[str(target_id)] = {"socket": client_socket, "id": container_id}
            lock.release()
            #client_socket.close()

        elif event == 'finished':
            lock.acquire()
            container_id = str(container_id)
            if container_id in waiting_cnt_sockets:
                print('Server - container {} finished working, '
                      'notifying source container {}'.format(container_id, waiting_cnt_sockets[container_id]["id"]))
                waiting_cnt_socket = waiting_cnt_sockets[container_id]["socket"]
                waiting_cnt_socket.sendall('Success'.encode())
                waiting_cnt_socket.shutdown(socket.SHUT_RDWR)
                waiting_cnt_socket.close()
            lock.release()
            client_socket.shutdown(socket.SHUT_RDWR)
            client_socket.close()
        else:
            client_socket.shutdown(socket.SHUT_RDWR)
            client_socket.close()
            raise ValueError()

if __name__ == "__main__":
# rerad in CMD arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--vlab", help = "Path to Directory on host containing VirtualLab (default is assumed to be curent working directory).", default=os.getcwd())
    parser.add_argument("-i", "--Run_file", help = "Runfile to use (default is assumed to be curent working directory).", default="Run.py")
    parser.add_argument("-D", "--Docker", help="Flag to use docker on Linux host instead of defaulting to Singularity. \
                         This will be ignored on Mac/Windows as Docker is the default.",action='store_true')

    args = parser.parse_args()
    vlab_dir=os.path.abspath(args.vlab)
# Set flag to allow cmd switch between singularity and docker when using linux host.
    use_singularity = check_platform() and not args.Docker
    Run_file = args.Run_file
    # start server listening for incoming jobs on seperate thread
    lock = threading.Lock()
    thread = threading.Thread(target=process,args=(vlab_dir,use_singularity))
    thread.daemon = True


    thread.start()
    #start VirtualLab
    lock.acquire()
    if use_singularity:
        proc=subprocess.Popen('singularity exec --no-home --writable-tmpfs --nv -B /usr/share/glvnd -B {}:/home/ibsim/VirtualLab VL_GVXR.sif '
                        'VirtualLab -f /home/ibsim/VirtualLab/RunFiles/{}'.format(vlab_dir,Run_file), shell=True)
    else:
        # Assume using Docker
        proc=subprocess.Popen('docker run -it -v {}:/home/ibsim/VirtualLab ibsim/base '
                        'VirtualLab -f /home/ibsim/VirtualLab/RunFiles/{}'.format(vlab_dir,Run_file), shell=True)

    lock.release()
    # wait untill virtualLab is done before closing
    proc.wait()
