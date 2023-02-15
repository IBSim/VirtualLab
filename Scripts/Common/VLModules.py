import sys

sys.dont_write_bytecode = True
import atexit
import copy
import datetime
import json
import os
import shutil
import dill
from importlib import import_module
from types import SimpleNamespace as Namespace
from importlib import import_module, reload
import VLconfig
from .VirtualLab import VLSetup
from .VLContainer import Container_Utils as Utils
from . import VLFunctions as VLF
from importlib import import_module

###############################################################################
######################     base module class     ##############################
###############################################################################
DefaultSettings = {
    "Mode": "H",
    "Launcher": "Process",
    "NbJobs": 1,
    "Max_Containers": 1,
    "InputDir": VLconfig.InputDir,
    "OutputDir": VLconfig.OutputDir,
    "MaterialDir": VLconfig.MaterialsDir,
    "Cleanup": True,
}


class VLModule(VLSetup):
    def __init__(self, Simulation, Project, Cont_id=2, debug=False):
        # perform setup steps that are common to both VLModule and VLSetup
        self._Common_init(Simulation, Project, DefaultSettings, Cont_id, debug)

    def start_module(self):
        import threading

        # send ready message then wait to receive runs to perform from the server
        ready_msg = {"msg": "Ready", "Cont_id": self.Container}
        Utils.send_data(self.tcp_sock, ready_msg)
        while True:
            data = Utils.receive_data(self.tcp_sock, self.debug)

            if data:
                if data["msg"] == "Container_runs":
                    self.Logger(
                        f"container {self.Container} received job list from server.",
                        print=True,
                    )
                    # full_task_list = data['tasks']
                    self.run_list = data["tasks"]
                    self.settings_dict = data["settings"]
                    self.Settings(**self.settings_dict)
                    self.run_args = data["run_args"]
                    self.method_name = data["Method"]
                    self.dry_run = data["dry_run"]
                    break

        # create dictionary of parameters associated with the method_name
        # from the parameter file(s) using the namespace defined in the
        # method config file.
        method_cls = getattr(self, self.method_name)
        VLnamespace = self.method_config[self.method_name]["Namespace"]
        method_dicts = self._CreateParameters(VLnamespace)
        if self.dry_run:
            print(f"Performing dry run of {data['Method']}")
            method_cls.SetFlag(False)
        method_cls._MethodSetup(method_dicts)

        # Start heartbeat thread to message back to the server once every n seconds
        thread = threading.Thread(target=self.heartbeat, args=())
        thread.daemon = True
        thread.start()
        return

    def start_module2(self):
        import threading

        # send ready message then wait to receive runs to perform from the server
        ready_msg = {"msg": "Ready", "Cont_id": self.Container}
        Utils.send_data(self.tcp_sock, ready_msg)
        while True:
            data = Utils.receive_data(self.tcp_sock, self.debug)
            print(data)
            if data:
                if data["msg"] == "Container_runs":
                    self.Logger(
                        f"container {self.Container} received job list from server.",
                        print=True,
                    )
                    # full_task_list = data['tasks']
                    self.run_list = data["tasks"]
                    self.run_args = data["run_args"]
                    self.method_name = data["Method"]
                    self.dry_run = data["dry_run"]
                    break

        # create dictionary of parameters associated with the method_name
        # from the parameter file(s) using the namespace defined in the
        # method config file.
        method_cls = getattr(self, self.method_name)
        VLnamespace = self.method_config[self.method_name]["Namespace"]
        method_dicts = self._CreateParameters(VLnamespace)
        if self.dry_run:
            print(f"Performing dry run of {data['Method']}")
            method_cls.SetFlag(False)
        method_cls._MethodSetup(method_dicts)

        # Start heartbeat thread to message back to the server once every n seconds
        thread = threading.Thread(target=self.heartbeat, args=())
        thread.daemon = True
        thread.start()
        return

    def get_args(self):
        """
        function to get the arguments for inputting into the call to the method.
        """
        return self.run_args

    def heartbeat(self, heart_rate=5):
        """
        Function to send a message periodically to tell the server
        that the Module is still running. optional argument heart_rate
        sets the number of seconds between each message.

        """
        import time

        beat_msg = {"msg": "Beat", "Cont_id": self.Container}
        while True:
            Utils.send_data(self.tcp_sock, beat_msg)
            time.sleep(heart_rate)

    def filter_runs(self, param_dict):
        """
        Function to extract subset of runs from a
        parm_dict.

        Inputs:

        - param_dict: dictionary generated by create
                      parameters function.

        """
        # if given a subset of runs extract only those runs
        if self.run_list:
            all_runs = list(param_dict.keys())
            run_list = [all_runs[i] for i in self.run_list]
        else:
            run_list = list(param_dict.keys())

        param_dict = {key: param_dict[key] for key in param_dict.keys() & run_list}

        return param_dict

    def _Cleanup(self, Cleanup=True):

        exitstr = (
            "\n#############################\n"
            "###  Container Terminated ###\n"
            "#############################\n"
        )
        if not Cleanup:
            exitstr = (
                "The temp directory {} has not been deleted.\n".format(self.TEMP_DIR)
                + exitstr
            )
        elif os.path.isdir(self.TEMP_DIR):
            shutil.rmtree(self.TEMP_DIR)
        Utils.Cont_Finished(self.Container, self.tcp_sock)
        print(exitstr)

    def Parameters(
        self,
        Parameters_Master,
        Parameters_Var=None,
        ParameterArgs=None,
        Import=False,
        **run_flags,
    ):
        flags = {
            "Run{}".format(name): True for name in self.Methods
        }  # all defaulted to True
        super().Parameters(
            Parameters_Master, Parameters_Var, ParameterArgs, Import, **run_flags
        )
        self.start_module()
        #         # call setup for each method
        # for method_name in self.Methods:
        #     # get method_name instance
        #     method_cls = getattr(self,method_name)
        #     # create dictionary of parameters associated with the method_name
        #     # from the parameter file(s) using the namespace defined in the
        #     # method config file.
        #     VLnamespace = self.method_config[method_name]['Namespace']
        #     method_dicts = self._CreateParameters(VLnamespace)
        #     # add flag to the instance
        #     method_cls.SetFlag(flags['Run{}'.format(method_name)])
        #     method_cls._MethodSetup(method_dicts)
        
        
        
        
class VLModule2():
    def __init__(self, VL, debug=False):

        sock = Utils.create_tcp_socket()

        VL.tcp_sock = sock
        
        self.VL = VL
        
        self.tcp_sock = sock
        self.Container = VL.Container
        self.Logger = VL.Logger
        self.debug = debug

        self.start_module() # initiate things
        
        method = self.get_method() # get from data received
        method_inst = getattr(VL,method)
        method_inst.clsname = 'VLModule'
        
    def start_module(self):
        import threading

        # send ready message then wait to receive runs to perform from the server
        ready_msg = {"msg": "Ready", "Cont_id": self.Container}
        Utils.send_data(self.tcp_sock, ready_msg)
        while True:
            data = Utils.receive_data(self.tcp_sock, self.debug)
            if data:
                if data["msg"] == "Container_runs":
                    self.Logger(
                        f"container {self.Container} received job list from server.",
                        print=True,
                    )
                    self.data = data
                    break

        # Start heartbeat thread to message back to the server once every n seconds
        thread = threading.Thread(target=self.heartbeat, args=())
        thread.daemon = True
        thread.start()
        return

    def get_args(self):
        """
        function to get the arguments for inputting into the call to the method.
        """
        return self.data["run_args"]

    def get_method(self):
        return self.data["Method"]
        
    def heartbeat(self, heart_rate=5):
        """
        Function to send a message periodically to tell the server
        that the Module is still running. optional argument heart_rate
        sets the number of seconds between each message.

        """
        import time

        beat_msg = {"msg": "Beat", "Cont_id": self.Container}
        while True:
            Utils.send_data(self.tcp_sock, beat_msg)
            time.sleep(heart_rate)


    def Run(self):
        method = self.get_method()
        method_inst = getattr(self.VL,method)
        args = self.get_args()
        method_inst(**args)
        
    def Terminate(self):
        Utils.Cont_Finished(self.VL.Container, self.VL.tcp_sock)

        
    
        
 
