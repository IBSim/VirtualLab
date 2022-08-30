import socket
import json
from types import SimpleNamespace as Namespace
def RunJob(Cont_id,Tool,Parameters_Master,Parameters_Var,Project,Simulation):
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
            "Parameters_Var":Parameters_Var,"Project":Project,"Simulation":Simulation}
    # Long Note: we are fully expecting Parameters_Master and Parameters_Var to be strings 
    # pointing to Runfiles. However base VirtualLab supports passing in Namespaces.
    # (see virtualLab.py line 178 and GetParams for context). 
    # For the sake of my sanity we assume you are using normal RunFiles, which most users 
    # likley are.
    #
    # Therefore this check is here to catch any determined soul and let you know if you 
    # want to pass in Namespaces for container tools you will need implement it yourself. 
    # In principle this means convering Namespace to a dict with vars(Parameters_Master).
    # Then sending it over as json string and convering it back on the otherside.
    # In practrice you may run into issues with buffer sizes for sock.recv as the strings
    # can get very long indeed.
    if isinstance(Parameters_Master,Namespace):
        raise Exception("Passing in Namespaces is not currently supported for Container tools. \
        These must be strings pointing to a Runfile.")
    
    data_string = json.dumps(data)
    sock = socket.socket()
    # send a signal to VL_server saying you want to run a CIL container
    sock.connect(("0.0.0.0", 9999))
    sock.sendall(data_string.encode('utf-8'))
    data = sock.recv(1024).decode('utf-8') #wait to recive message saying the tool is finished before continuing
    sock.close()
    return

def Cont_Started(Cont_id):
    ''' Function to send a Message to the main script to say the container has started.'''
    data = {"msg":"started","Cont_id":Cont_id}
    data_string = json.dumps(data)
    sock = socket.socket()
    sock.connect(("0.0.0.0", 9999))
    sock.sendall(data_string.encode('utf-8'))
    sock.close()
    return

def Cont_Finished(Cont_id):
    ''' Function to send a Message to the main script to say the container has Finished.'''
    data = {"msg":"finished","Cont_id":Cont_id}
    data_string = json.dumps(data)
    sock = socket.socket()
    sock.connect(("0.0.0.0", 9999))
    sock.sendall(data_string.encode('utf-8'))
    sock.close()
    return