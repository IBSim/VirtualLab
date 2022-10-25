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

def runs_to_file(filename):
    pass

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
        for proc in process_list.values():
            # check if the process has finished
            # Note: it may be a good idea to add 
            # a heartbeat check to guard against 
            # processes that just hang.
            try:
                outs, errs = proc.communicate(timeout=1)
            except TimeoutExpired :
                continue
            #communicate sets returncode inside proc if finished
            if proc.returncode is not None and proc.returncode != 0 :      
            #    except TimeoutExpired :
            #        continue
            #poll sets returncode inside proc if finished
            if proc.returncode != 0 :      
                #This converts the strings from bytes to utf-8 
                # however, we need to check they exist because
                # none objects can't be converted to utf-8
                if outs:
                    outs = str(outs,'utf-8')
                if errs:
                    errs = str(errs,'utf-8')
                
                err_mes = ContainerError(outs,errs)
                print(err_mes)
                sock_lock.acquire()
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
                #client_socket.shutdown(socket.SHUT_RDWR)
                #client_socket.close()
                
    return

waiting_cnt_sockets = {}
running_processes={}
target_ids = []
task_dict = {}

def process(vlab_dir,use_singularity):
    sock_lock = threading.Lock()
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,1)
    sock.bind(("0.0.0.0", 9999))
    next_cnt_id = 2
    while True:
    sock.listen(20)
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
                container_cmd = 'docker run --rm -it --user=$(id -u $USER):$(id -g $USER)'\
                                '--env="DISPLAY" \--volume="/etc/group:/etc/group:ro"' \
                                '--volume="/etc/passwd:/etc/passwd:ro"' \
                                '--volume="/etc/shadow:/etc/shadow:ro"' \
                                '--volume="/etc/sudoers.d:/etc/sudoers.d:ro"' \
                                '--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw"'

            sock_lock.acquire()
            
            # loop over containers once to create a dict of final container ids
            # and associated runs to output to file
            for Container in Cont_runs:
                target_ids.append(next_cnt_id)
                list_of_runs = Container[1]
                task_dict[str(next_cnt_id)] = list_of_runs
                next_cnt_id += 1

            # loop over containers again to spawn them this time
            for n,Container in enumerate(Cont_runs):    
                options, command = Format_Call_Str(tool,vlab_dir,param_master,
                    param_var,project,simulation,use_singularity,target_ids[n])

                print(f'Server - starting a new container with ID: {target_ids[n]} '
                      f'as requested by container {container_id}')

                # spawn the container
                container_process = subprocess.Popen(f'{container_cmd} {options} {command}',
                    stderr = subprocess.PIPE, shell=True)
                
                waiting_cnt_sockets[str(target_ids[n])] = {"socket": client_socket, "id": container_id}

                running_processes[str(target_ids[n])] = container_process
                
                # send message to tell client container what id the new container will have
                data = {"msg":"Running","Cont_id":target_ids[n]}
                send_data(client_socket, data)
            print('hit')
            sock_lock.release()
            #continue

        elif event == 'Ready':
            sock_lock.acquire()
            # finally send the list of tasks and id's to the containers
            data2 = {"msg":"Container_runs","tasks":task_dict}
            send_data(client_socket, data2)
            sock_lock.release()

        elif event == 'Finished':
            sock_lock.acquire()
            container_id = str(container_id)
            if container_id in waiting_cnt_sockets:
                print(f'Server - container {container_id} finished working, '
                      f'notifying source container {waiting_cnt_sockets[container_id]["id"]}')
                waiting_cnt_socket = waiting_cnt_sockets[container_id]["socket"]
                data = {"msg":"Success","Cont_id":waiting_cnt_sockets[container_id]["id"],"target_id":int(container_id)}
                send_data(waiting_cnt_socket,data)
                running_processes.pop(str(container_id))
            sock_lock.release()
            
        else:
            client_socket.shutdown(socket.SHUT_RDWR)
            client_socket.close()
            raise ValueError(f'Unknown message {event} received')
        print("hit")
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
        # start server listening for incoming jobs on separate thread
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
        # wait until virtualLab is done before closing
        proc.wait()
    else:
        # use native version
        print("Warning: VirtualLab is Running in native mode Some tools will be unavalible.\n"
            " Since version 2.0 VirtualLab is being setup to run inside a container with \n "
            " Docker/Singularity. To use this container functionality, and tools which depend \n"
            " on it you will need to install docker or Singularity and pass in the -C option. \n")

        proc=subprocess.check_call(f'VirtualLab -f {Run_file}', shell=True)





