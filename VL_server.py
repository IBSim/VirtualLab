import socket
import sys
import subprocess
import threading
import argparse
import os

next_cnt_id = 1
waiting_cnt_sockets = {}

# watch -n1 ps aux | grep 'Apptainer runtime'

def process(job_done):
    global next_cnt_id
    lock = threading.Lock()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 9999))
    sock.listen(20)

    while True:
        client_socket, client_address = sock.accept()
        data = client_socket.recv(1024).decode()
        event, container_id = data.split(':')

        print('Server - received "{}" event from container {} at {}'.format(event, container_id, client_address))

        if event == 'VirtualLab started':
            client_socket.close()
        elif event == 'runJob':
            lock.acquire()
            
            print('Server - starting a new container with ID: {} '
                  'as requested by container {}'.format(next_cnt_id, container_id))

            subprocess.Popen('singularity exec --bind ./cont2/:/cont2/ python_3.8-slim.sif  '
                             'python /cont2/cont2_client.py {}'.format(next_cnt_id), shell=True)

            waiting_cnt_sockets[str(next_cnt_id)] = {"socket": client_socket, "id": container_id}
            next_cnt_id += 1

            lock.release()

        elif event == 'VirtualLab finished':
            lock.acquire()
            if container_id in waiting_cnt_sockets:
                print('Server - container {} finished working, '
                      'notifying source container {}'.format(container_id, waiting_cnt_sockets[container_id]["id"]))
                waiting_cnt_socket = waiting_cnt_sockets[container_id]["socket"]
                waiting_cnt_socket.sendall('Success'.encode())
                waiting_cnt_socket.close()
            lock.release()
            client_socket.close()
            #set flag to tell master thread its done and exit thread
            job_done.set()
        else:
            raise ValueError()
if __name__ == "__main__":
# rerad in CMD arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--vlab", help = "Path to Directory ion host containing VirtualLab (default is assumed to be curent working directory).", default=os.getcwd())
    parser.add_argument("-i", "--Run_file", help = "Runfile to use (default is assumed to be curent working directory).", default="Run.py")
    args = parser.parse_args()
    # start server listening for incoming jobs on seperate thread
    job_done = threading.Event()
    lock = threading.Lock()
    thread = threading.Thread(target=process,args=(job_done,))
    thread.daemon = True

    vlab_dir=os.path.abspath(args.vlab)
    Run_file = args.Run_file
    thread.start()
    #start VirtualLab
    lock.acquire()
    subprocess.Popen(f'SINGULARITYENV_PYTHONPATH=/home/ibsim/VirtualLab/third_party/GVXR_Install/gvxrWrapper-1.0.5/python3 singularity exec --contain --writable-tmpfs --env-file contain_envs -B {vlab_dir}:/home/ibsim/VirtualLab virtualLab.sif '
                    f'VirtualLab -f /home/ibsim/VirtualLab/RunFiles/{Run_file}', shell=True)
    next_cnt_id += 1
    lock.release()
    # wait untill all threads and virtualLab are done before closing
    job_done.wait()
