'''
        Script to enable comunication with and spawning of containers.
        #######################################################################
        Note: Current container ID's are:
        1 - Base VirtualLab
        2 - CIL
        3 - GVXR
        4 - Container tests

        If/when you want to add more containers you will need to give 
        them a unique id in the container_id's dict in this file 
        and add it to this list as a courtesy so everyone is on the same page
        #######################################################################
'''
import socket
import subprocess
import threading
import argparse
import os
import json
from Scripts.Common.VLContainer.container_tools import check_platform,Format_Call_Str

waiting_cnt_sockets = {}

Container_IDs = {"VLab":1,"CIL":2,"GVXR":3,"Test":4}
def process(vlab_dir,use_singularity):
    sock_lock = threading.Lock()
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
        print('Server - received "{event}" event from container {container_id} at {client_address}')

        if event == 'VirtualLab started':
            client_socket.close()
        elif event == 'RunJob':
            tool = rec_dict["Tool"]
            param_master = rec_dict["Parameters_Master"]
            param_var = rec_dict["Parameters_Var"]
            project = rec_dict["Project"]
            simulation = rec_dict["Simulation"]

            sock_lock.acquire()
            options, command = Format_Call_Str(tool,vlab_dir,param_master,
            param_var,project,simulation,use_singularity)
            target_id = Container_IDs[tool]
            print(f'Server - starting a new container with ID: {target_id} '
                  'as requested by container {container_id}')
            # setup comand to run docker or singularity
            if use_singularity:
                container_cmd = 'singularity exec --contain --writable-tmpfs'
            else:
                container_cmd = 'docker run -it'

            try:
                proc = subprocess.check_call(f'{container_cmd} {options} {command}',
                     shell=True)
            except Exception:
                lock.release()
                client_socket.shutdown(socket.SHUT_RDWR)
                client_socket.close()
                raise
            waiting_cnt_sockets[str(target_id)] = {"socket": client_socket, "id": container_id}
            lock.release()
            #client_socket.close()

        elif event == 'finished':
            sock_lock.acquire()
            container_id = str(container_id)
            if container_id in waiting_cnt_sockets:
                print(f'Server - container {container_id} finished working, '
                      f'notifying source container {waiting_cnt_sockets[container_id]["id"]}')
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
    parser.add_argument("-d", "--vlab", help = "Path to Directory on host containing \
     VirtualLab (default is assumed to be curent working directory).", default=os.getcwd())
    parser.add_argument("-f", "--Run_file", help = "Runfile to use (default is assumed to \
        be curent working directory).", default="Run.py")
    parser.add_argument("-D", "--Docker", help="Flag to use docker on Linux host instead of \
        defaulting to Singularity.This will be ignored on Mac/Windows as Docker is the default.",
        action='store_true')
    parser.add_argument("-C", "--Container", help="Flag to use tools in Containers.",
                        action='store_true')

    args = parser.parse_args()
    vlab_dir=os.path.abspath(args.vlab)
# Set flag to allow cmd switch between singularity and docker when using linux host.
    use_singularity = check_platform() and not args.Docker
    Run_file = args.Run_file
    Container = args.Container
    if Container:
        # start server listening for incoming jobs on seperate thread
        lock = threading.Lock()
        thread = threading.Thread(target=process,args=(vlab_dir,use_singularity))
        thread.daemon = True


        thread.start()
        #start VirtualLab
        lock.acquire()
        if use_singularity:
            proc=subprocess.Popen(f'singularity exec --no-home --writable-tmpfs --nv -B \
                            /usr/share/glvnd -B {vlab_dir}:/home/ibsim/VirtualLab Containers/VL_GVXR.sif '
                            'VirtualLab -f /home/ibsim/VirtualLab/RunFiles/{Run_file}', shell=True)
        else:
            # Assume using Docker
            proc=subprocess.Popen(f'docker run -it -v {vlab_dir}:/home/ibsim/VirtualLab ibsim/base '
                            'VirtualLab -f /home/ibsim/VirtualLab/RunFiles/{Run_file}', shell=True)

        lock.release()
        # wait untill virtualLab is done before closing
        proc.wait()
    else:
        # use native version
        print("Warning: VirtualLab is Running in native mode Some tools will be unavalible.\n"
            " Since version 2.0 VirtualLab is being setup to run inside a container with \n "
            " Docker/Singularity. To use this container functionality, and tools which depend \n"
            " on it you will need to install docker or Singularity and pass in the -C option. \n")

        proc=subprocess.check_call(f'VirtualLab -f {Run_file}', shell=True)