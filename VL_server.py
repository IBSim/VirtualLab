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
from Scripts.Common.VLContainer.Container_Utils import check_platform, Format_Call_Str, send_data, receive_data 
def ContainerError(out,err):
    '''Custom function to format error message in a pretty way.'''
    Errmsg = "\n========= Container returned non-zero exit code =========\n\n"\
                f"std out: {out}\n\n"\
                f"std err:{err}\n\n"\
                "==================================\n\n"
    return Errmsg

def check_for_errors(process_list,client_socket,sock_lock):
    ''' 
    Function to take in nested a dictionary of containing running processes and container id's.
    Idealy any python errors will be handled and cleanup should print an error message to the screen
    and send a success message to the main process. Thus avoiding hanging the aplication.
    This function here is to catch any non-python errors. By simply running proc.communicate()
    to check each running process. From there if the return code is non zero it stops the server and
    spits out the std_err from the process.
     '''
    
    if not process_list:
    # if list is empty return straightway and continue in while loop.
        return
    else:
        for proc in process_list.values():
            outs, errs = proc.communicate()
            #communcatte sets retuncode inside proc
            if proc.returncode != 0 :
                
                #This convets the strings from bytes to utf-8 
                # however, we need to check they exist because
                # none objects can't be coverted to utf-8
                if outs:
                    outs = str(outs,'utf-8')
                if errs:
                    errs = str(errs,'utf-8')
                
                err_mes = ContainerError(outs,errs)
                print(err_mes)
                sock_lock.acquire()
                # send mesage to tell main vlab thread to close because there was an error 
                data = {"msg":"Error","stderr":'-1'}
                send_data(client_socket,data)
                sock_lock.release()
                client_socket.shutdown(socket.SHUT_RDWR)
                client_socket.close()
                
    return

waiting_cnt_sockets = {}
running_processes={}
def process(vlab_dir,use_singularity):
    sock_lock = threading.Lock()
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,1)
    sock.bind(("0.0.0.0", 9999))
    sock.listen(20)
    next_cnt_id = 2
    while True:
        client_socket, client_address = sock.accept()
        rec_dict = receive_data(client_socket)
        event = rec_dict["msg"]
        container_id = rec_dict["Cont_id"]
        print(f'Server - received "{event}" event from container {container_id} at {client_address}')

        if event == 'VirtualLab started':
            client_socket.close()
        elif event == 'RunJob':
            tool = rec_dict["Tool"]
            num_containers = rec_dict["Num_Cont"]
            param_master = rec_dict["Parameters_Master"]
            param_var = rec_dict["Parameters_Var"]
            project = rec_dict["Project"]
            simulation = rec_dict["Simulation"]
            sock_lock.acquire()
            target_id = next_cnt_id
            next_cnt_id += 1
            options, command = Format_Call_Str(tool,vlab_dir,param_master,
                param_var,project,simulation,use_singularity,target_id)

            print(f'Server - starting a new container with ID: {target_id} '
                  f'as requested by container {container_id}')

            # setup comand to run docker or singularity
            if use_singularity:
                container_cmd = 'singularity exec --writable-tmpfs'
            else:
                # this monstrosity logs the user in as "themself" to allow safe access top x11 graphical apps"
                #see http://wiki.ros.org/docker/Tutorials/GUI for more details
                container_cmd = 'docker run --rm -it --user=$(id -u $USER):$(id -g $USER)'\
                                '--env="DISPLAY" \--volume="/etc/group:/etc/group:ro"' \
                                '--volume="/etc/passwd:/etc/passwd:ro"' \
                                '--volume="/etc/shadow:/etc/shadow:ro"' \
                                '--volume="/etc/sudoers.d:/etc/sudoers.d:ro"' \
                                '--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw"'
            #try:
            container_process = subprocess.Popen(f'{container_cmd} {options} {command}',
                 stderr = subprocess.PIPE, shell=True)
            #except Exception:
            #    sock_lock.release()
            #    client_socket.shutdown(socket.SHUT_RDWR)
            #    client_socket.close()
            #    raise
            waiting_cnt_sockets[str(target_id)] = {"socket": client_socket, "id": container_id}

            running_processes[str(target_id)] = container_process
            # send mesage to tell client continer what id the new container will have
            data = {"msg":"Running","Cont_id":waiting_cnt_sockets[str(target_id)]["id"]}
            send_data(client_socket, data)
            sock_lock.release()
            #client_socket.close()

        elif event == 'Finished':
            sock_lock.acquire()
            container_id = str(container_id)
            if container_id in waiting_cnt_sockets:
                print(f'Server - container {container_id} finished working, '
                      f'notifying source container {waiting_cnt_sockets[container_id]["id"]}')
                waiting_cnt_socket = waiting_cnt_sockets[container_id]["socket"]
                data = {"msg":"Success","Cont_id":waiting_cnt_sockets[container_id]["id"]}
                send_data(waiting_cnt_socket,data)
                waiting_cnt_socket.shutdown(socket.SHUT_RDWR)
                waiting_cnt_socket.close()
                running_processes.pop(str(container_id))
            sock_lock.release()
            client_socket.shutdown(socket.SHUT_RDWR)
            client_socket.close()
        else:
            client_socket.shutdown(socket.SHUT_RDWR)
            client_socket.close()
            raise ValueError()
        check_for_errors(running_processes,client_socket,sock_lock)
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
                            f'VirtualLab -f /home/ibsim/VirtualLab/RunFiles/{Run_file}', shell=True)
        else:
            # Assume using Docker
            proc=subprocess.Popen(f'docker run -it -v {vlab_dir}:/home/ibsim/VirtualLab ibsim/base '
                            f'VirtualLab -f /home/ibsim/VirtualLab/RunFiles/{Run_file}', shell=True)

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





