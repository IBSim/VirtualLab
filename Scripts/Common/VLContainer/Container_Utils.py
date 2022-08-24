import socket
import json
def RunJob(Cont_id,Tool,Parameters_Master,Parameters_Var):
    ''' Function to enable comunication with host script from container.
        This will be called from the VirtualLab container to Run a job 
        with another toll in seperate contianer. At the moment this 
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
                deliberatly left in to allow other containers to call 
                tools should the need arise.

        Tool (str): String containg name of the tool you want to run (curently just CIL).

        Parameters_Master/Var (str): path INSIDE THE CONTAINER to the parmaeters file
        to be read in. Note you will need to setup directory binding carefully in VLsever.py.
    '''
        # setup networking to comunicate with host script whilst running in a continer
    data = {"msg":"RunJob","Cont_id":Cont_id,"Tool":Tool,"Parameters_Master":Parameters_Master,
            "Parameters_Var":Parameters_Var}

    data_string = json.dumps(data)
    sock = socket.socket()
    # send a signal to VL_server sa=ying you want to run a CIL container
    sock.connect(("127.0.0.1", 9999))
    sock.sendall(data_string.encode('utf-8'))
    sock.recv(1024) #wait to recive message saying the tool is finished before continuing
    sock.close()
    return

def Cont_Started(Cont_id):
    ''' Function to send a Message to the main script to say the container has started.'''
    data = {"msg":"started","Cont_id":Cont_id}
    data_string = json.dumps(data)
    sock = socket.socket()
    sock.connect(("127.0.0.1", 9999))
    sock.sendall(data_string.encode('utf-8'))
    sock.close()
    return

def Cont_Finished(Cont_id):
    ''' Function to send a Message to the main script to say the container has Finished.'''
    data = {"msg":"finished","Cont_id":Cont_id}
    data_string = json.dumps(data)
    sock = socket.socket()
    sock.connect(("127.0.0.1", 9999))
    sock.sendall(data_string.encode('utf-8'))
    sock.close()
    return