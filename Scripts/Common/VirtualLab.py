#!/usr/bin/env python3

import sys

sys.dont_write_bytecode = True
import datetime
import os
import shutil
import copy
from types import SimpleNamespace as Namespace, ModuleType
import atexit
import json
import numpy as np
import uuid
import VLconfig
from .VLContainer import Container_Utils as Utils
from . import VLFunctions as VLF
from importlib import import_module

#############################################################
# Note: VLSetup is the VLManager in the V2.0 naming scheme. #
#       However, changing it's name would break far to much #
#       legacy code to be worthwhile. So it's easier to     #
#       just live with it for now.                          #
#############################################################
DefaultSettings = {
    "Mode": "H",
    "Launcher": "Process",
    "NbJobs": 1,
    "Max_Containers": 1,
    "InputDir": VLconfig.InputDir,
    "OutputDir": VLconfig.OutputDir,
    "MaterialDir": VLconfig.MaterialsDir,
    "Cleanup": True,
    "dry_run": False,
    "debug": False,
    "tcp_port": 9000,
}


class VLSetup:
    def __init__(self, Simulation, Project, Cont_id=1):
        self._parsed_kwargs = VLF.parsed_kwargs(
            sys.argv[1:]
        )  # may need to be more robust than sys.argv
        # perform setup steps that are common to both VLModule and VL_manger
        self._Common_init(Simulation, Project, DefaultSettings, Cont_id)
        # Unique ID
        git_id = self._git()
        self._ID = "{}_{}".format(git_id, self._time)
        data = {"msg": "VirtualLab started", "Cont_id": 1}
        data_string = json.dumps(data)
        Utils.send_data(self.tcp_sock, data)
        self.Logger(
            "\n############################\n"
            "### Launching VirtualLab ###\n"
            "############################\n",
            Print=True,
        )

    def __getstate__(self):
        """
        This is here to solve issues with pickling when using mpi.
        Specifically because we pass a VLsetup/VLmodule object into
        the call to mpi. When this occurs the class gets serialized through pickle
        to be sent to each mpi process. The problem is not all objects in the class can be serialized.

        This dundder method thus provides a workaround since __getstate__ gets called before pickling.
        Thus we can remove the offending attributes since they are not needed by the mpi processes.

        Note: we do  not directly modify self.__dict__ but instead copy it and serialise the copy. This is
        because we dont want to modify the object itself but instead send a modified version to each mpi process.

        """
        attributes = self.__dict__.copy()
        attributes.pop("tcp_sock", None)
        return attributes

    def AddToPath(self, path, ix=-1):
        """A more robust way of adding paths as they will also be available inside
        every container"""
        sys.path.insert(ix, path)
        if not hasattr(self, "_AddedPaths"):
            self._AddedPaths = []
        self._AddedPaths.append(path)

    def _AddMethod(self):
        """Add in the methods defined in Scripts/Methods to the VirtualLab class."""
        MethodsDir = "{}/Methods".format(self.SCRIPTS_DIR)
        self.Methods = []
        self.method_config = self.load_config(self.CONF_DIR, "VL_Methods.json")
        # Loop through directory contents
        for method_name in self.method_config.keys():
            # skip directiories, files that start with '_' and those that aren't python
            if method_name.startswith("_"):
                continue
            if not os.path.isfile("{}/{}.py".format(MethodsDir, method_name)):
                continue

            # define the path to the scripts for a certain method & add to class
            script_path = "{}/{}".format(self.SIM_SCRIPTS, method_name)
            setattr(self, "SIM_{}".format(method_name.upper()), script_path)

            # If there's a config.py file in the methods directory this is used instead
            if os.path.isfile("{}/config.py".format(script_path)):
                mod_path = "{}.config".format(method_name)
            else:
                mod_path = "Scripts.Methods.{}".format(method_name)
            method_mod = import_module(mod_path)
            # Try and import the method
            # try:
            #     method_mod = import_module(mod_path)
            # except :
            #     print(VLF.WarningMessage("Error during import of method '{}'.\nThis method will be unavailable for analysis".format(method_name)))
            #     continue

            # check the imported method has a class called Method
            if not hasattr(method_mod, "Method"):
                self.Exit(
                    VLF.ErrorMessage(
                        "The method '{}' does not have the required class 'Method'".format(
                            method_name
                        )
                    )
                )

            # initiate class and wrap key function
            method_inst = method_mod.Method(self)

            # add the method to self and add to list of methods
            setattr(self, method_name, method_inst)
            self.Methods.append(method_name)

    def handle_except(self, *args):
        """
        This function stops errors in containers from occurring silently by
        catching them with sys.excepthook and printing a traceback to the
        user via VL.Logger. When python errors are raised it normally only
         prints the traceback to the the container not the host.
         Hence we need to be careful errors are handled appropriately.

        If this were not here Cleanup will get run Via AtExit.
        This sends the Finished message to the server to tell the main
        thread to close. This is what we want. However, AtExit does not
        distinguish between errors and a normal exit. More annoyingly
        it returns 0. Thus the server assumes all is well.

        This Function is a bit of a hack but at least it gives
        you a heads-up that all is not well.
        """
        import traceback as tb

        btrace = "".join(tb.format_exception(None, None, args[2]))
        errtype = str(args[1])
        errormsg = (
            "\n############################\n"
            "###  Error Occurred   ######\n"
            "###  in the container ######\n"
            f"{errtype}\n"
            f"{btrace}"
            "############################\n"
        )

        self.Logger(errormsg, Print=True)

    def load_config_yaml(self, vlab_dir, filename):
        """Function to get the config from a .yaml file"""
        import yaml
        from pathlib import Path

        vlab_dir = Path(vlab_dir)
        # load module config from yaml_file
        config_file = vlab_dir / filename
        with open(config_file) as file:
            try:
                config = yaml.safe_load(file)
            except yaml.YAMLError as exception:
                print(exception)
        return config

    def load_config(self, vlab_dir, filename):
        """Function to get the config from a json file"""
        import json
        from pathlib import Path

        vlab_dir = Path(vlab_dir)
        # load module config from file
        config_file = vlab_dir / filename
        with open(config_file) as file:
            config = json.load(file)
        return config

    def _Common_init(self, Simulation, Project, DefaultSettings, Cont_id=1):
        """
        init steps that are common between both VLSetup and VLModule.
        These are here since it makes sense to have them in one place and
        save duplicating work.
        """

        sys.excepthook = self.handle_except
        # ======================================================================
        # Check for updates to Simulation and Project in parsed arguments
        # Use given Simulation and project if not in self._parsed_kwargs
        self.Simulation = self._parsed_kwargs.get("Simulation", Simulation)
        self.Project = self._parsed_kwargs.get("Project", Project)
        # Copy path at the start for MPI to match sys.path
        self._pypath = sys.path.copy()
        self.Container = Cont_id
        # ======================================================================

        # ======================================================================
        # Define path to scripts
        # check simulation type exists
        self.SCRIPTS_DIR = "{}/Scripts".format(VLconfig.VL_DIR_CONT)

        self.SIM_SCRIPTS = "{}/Experiments/{}".format(self.SCRIPTS_DIR, self.Simulation)
        self.CONF_DIR = "{}/Config".format(VLconfig.VL_DIR_CONT)
        if not os.path.isdir(self.SIM_SCRIPTS):
            self.Exit(
                VLF.ErrorMessage(
                    "Simulation type '{0}' does not exist.\n"
                    "Please check you have created a directory named '{0}' "
                    "inside the Scripts/Experiments directory.".format(self.Simulation)
                )
            )
        self.COM_SCRIPTS = "{}/Common".format(self.SCRIPTS_DIR)
        self.VLRoutine_SCRIPTS = "{}/VLRoutines".format(self.COM_SCRIPTS)
        # Add these to path
        for path in [self.SCRIPTS_DIR, self.COM_SCRIPTS, self.SIM_SCRIPTS]:
            self.AddToPath(path)

        # =======================================================================
        # Define & create temporary directory for work to be saved to
        self._TempDir = VLconfig.TEMP_DIR
        # timestamp
        self._time = (datetime.datetime.now()).strftime("%y.%m.%d_%H.%M.%S.%f")

        self.TEMP_DIR = "{}/VL_{}".format(self._TempDir, self._time)
        try:
            os.makedirs(self.TEMP_DIR)
        except FileExistsError:
            # Unlikely this would happen. Suffix random number to directory name
            self.TEMP_DIR = "{}_{}".format(self.TEMP_DIR, np.random.randint(1000))
            os.makedirs(self.TEMP_DIR)
        # Specify default settings
        self.Settings(**DefaultSettings)
        # create socket for networking
        self.tcp_sock = Utils.create_tcp_socket(self._tcp_port)
        self._AddMethod()

    def _SetMode(self, Mode="H"):
        # ======================================================================
        # Update mode as shorthand version can be given
        if Mode.lower() in ("i", "interactive"):
            self.mode = "Interactive"
        elif Mode.lower() in ("t", "terminal"):
            self.mode = "Terminal"
        elif Mode.lower() in ("c", "continuous"):
            self.mode = "Continuous"
        elif Mode.lower() in ("h", "headless"):
            self.mode = "Headless"
        else:
            self.Exit(
                VLF.ErrorMessage(
                    "Mode must be one of; 'Interactive',\
                                      'Terminal','Continuous', 'Headless'"
                )
            )

    def _SetLauncher(self, Launcher="Process"):
        if Launcher.lower() == "sequential":
            self._Launcher = "Sequential"
        elif Launcher.lower() == "process":
            self._Launcher = "Process"
        elif Launcher.lower() == "mpi":
            self._Launcher = "MPI"
        elif Launcher.lower() == "mpi_worker":
            self._Launcher = "MPI_Worker"
        else:
            self.Exit(
                VLF.ErrorMessage(
                    "Launcher must be one of; 'Sequential',\
                                     'Process', 'MPI'"
                )
            )

    def _SetNbJobs(self, NbJobs=1):
        if type(NbJobs) == int:
            _NbJobs = NbJobs
        elif type(NbJobs) == float:
            if NbJobs.is_integer():
                _NbJobs = NbJobs
            else:
                self.Exit(VLF.ErrorMessage("NbJobs must be an integer"))
        else:
            self.Exit(VLF.ErrorMessage("NbJobs must be an integer"))

        if _NbJobs >= 1:
            self._NbJobs = _NbJobs
        else:
            self.Exit(VLF.ErrorMessage("NbJobs must be positive"))

    def _SetInputDir(self, InputDir):
        if not os.path.isdir(InputDir):
            self.Exit(VLF.ErrorMessage("InputDir is not a valid directory"))
        self._InputDir = InputDir
        self.PARAMETERS_DIR = "{}/{}/{}".format(
            self._InputDir, self.Simulation, self.Project
        )

    def _SetOutputDir(self, OutputDir):
        self._OutputDir = OutputDir
        self.PROJECT_DIR = "{}/{}/{}".format(
            self._OutputDir, self.Simulation, self.Project
        )

    def _SetMaterialDir(self, MaterialDir):
        if not os.path.isdir(MaterialDir):
            self.Exit(VLF.ErrorMessage("MaterialDir is not a valid directory"))
        self.MATERIAL_DIR = MaterialDir

    def _SetCleanup(self, Cleanup=True):
        if not hasattr(self, "_CleanupFlag"):
            self._CleanupFlag = Cleanup
        else:
            pass
            atexit.unregister(self._Cleanup)
        atexit.register(self._Cleanup, Cleanup)

    def _SetMax_Containers(self, Max_Containers=1):
        self._Max_Containers = Max_Containers
        return

    def _SetTcp_Port(self, tcp_port=9000):
        import os

        if type(tcp_port) != int:
            self.Exit(
                VLF.ErrorMessage(
                    f"Invalid port number: {tcp_port}, must be an integer."
                )
            )
        elif tcp_port < 1024 or tcp_port > 65535:
            self.Exit(
                VLF.ErrorMessage(
                    f"invalid port number: {tcp_port}, must be an integer between 1024 and 65535."
                )
            )
        else:
            os.environ["VL_TCP_PORT"] = str(tcp_port)
            self._tcp_port = tcp_port
        return

    def _SetDryrun(
        self,
        dry_run=False,
    ):
        if type(dry_run) == bool:
            self._dry_run = dry_run
        elif str(dry_run).lower() in ["true", "false", "0", "1"]:
            self._dry_run = dry_run
        else:
            self.Exit(
                VLF.ErrorMessage(
                    f"Invalid option: {dry_run} for Dry_run. Must a boolean value"
                )
            )
        return

    def _SetDebug(
        self,
        debug=False,
    ):
        if type(debug) == bool:
            self._debug = debug
        elif str(debug).lower() in ["true", "false", "0", "1"]:
            self._debug = debug
        else:
            self.Exit(
                VLF.ErrorMessage(
                    f"Invalid option:{debug} for debug. Must a boolean value"
                )
            )
        return

    @VLF.kwarg_update
    def Settings(self, **kwargs):
        # Dont specify the kwarsg so that the defauls aren't overwritten if there
        # are multiple calls to settings
        # dictionary of available kwargs and the function used to specify them
        kwargs_fnc = {
            "Mode": self._SetMode,
            "Launcher": self._SetLauncher,
            "NbJobs": self._SetNbJobs,
            "InputDir": self._SetInputDir,
            "OutputDir": self._SetOutputDir,
            "MaterialDir": self._SetMaterialDir,
            "Cleanup": self._SetCleanup,
            "dry_run": self._SetDryrun,
            "debug": self._SetDebug,
            "tcp_port": self._SetTcp_Port,
            "Max_Containers": self._SetMax_Containers,
        }

        # check no incorrect kwargs given
        Diff = set(kwargs).difference(kwargs_fnc.keys())
        if Diff:
            self.Exit(
                VLF.ErrorMessage(
                    "The following are not valid options in Settings:\n{}".format(
                        "\n".join(Diff)
                    )
                )
            )

        updated_kwargs = VLF.Parser_update(kwargs_fnc.keys(), self._parsed_kwargs)
        kwargs.update(updated_kwargs)

        for kw_name, kw_fnc in kwargs_fnc.items():
            # if kw_name is in kwargs then we set it using kw_fnc
            if kw_name in kwargs:
                kw_fnc(kwargs[kw_name])
        self.settings_dict = kwargs

    @VLF.kwarg_update
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

        # check no incorrect kwargs given
        Diff = set(run_flags).difference(flags.keys())
        if Diff:
            self.Exit(
                VLF.ErrorMessage(
                    "The following are not valid options in Parameters:\n{}".format(
                        "\n".join(Diff)
                    )
                )
            )
        flags.update(run_flags)

        # update run_flags keywords (not covered by decorator as they depend on the files in methods directory)
        updated_flags = VLF.Parser_update(flags.keys(), self._parsed_kwargs)
        flags.update(updated_flags)

        # update Parameters_master with parser (not covered by decorator as its an argument)
        Parameters_Master = self._parsed_kwargs.get(
            "Parameters_Master", Parameters_Master
        )

        self._SetParams(Parameters_Master, Parameters_Var, ParameterArgs=ParameterArgs)
        # get the number of runs defined in params for each module
        self.Num_runs = self._get_Num_Runs(flags, self.Methods)
        # get a list of all the containers and the runs they will process for each module
        self.container_list = self._Spread_over_Containers()
        # call setup for each method
        for method_name in self.Methods:
            # get method_name instance
            method_cls = getattr(self, method_name)
            # add flag to the instance
            method_cls.SetFlag(flags["Run{}".format(method_name)])
            _NS_name = self.method_config[method_name]["Namespace"]
            method_dicts = self._CreateParameters(_NS_name)
            method_cls._MethodSetup(method_dicts)

    def ImportParameters(self, Rel_Parameters, ParameterArgs=None):
        """
        Rel_Parameters is a file name relative to the Input directory
        """
        # Strip .py off the end if it's in the name
        if os.path.splitext(Rel_Parameters)[1] == ".py":
            Rel_Parameters = os.path.splitext(Rel_Parameters)[0]

        Abs_Parameters = "{}/{}.py".format(self.PARAMETERS_DIR, Rel_Parameters)
        # Check File exists
        if not os.path.exists(Abs_Parameters):
            message = "The following Parameter file does not exist:\n{}".format(
                Abs_Parameters
            )
            self.Exit(VLF.ErrorMessage(message))

        if ParameterArgs != None:
            # If arguments are provided then it's pickled to a file
            arg_path = "{}/{}.pkl".format(self.TEMP_DIR, uuid.uuid4())
            VLF.WriteArgs(arg_path, ParameterArgs)
            sys.argv.append("ParameterArgs={}".format(arg_path))

        Parameters = VLF.GetModule(Abs_Parameters)

        if ParameterArgs != None:
            sys.argv.pop(-1)

        return Parameters

    def _SetParams(self, Master, Var, ParameterArgs=None):
        """Master & Var can be a module, namespace, string or None.
        A string references a file to import from within the input directory.
        """

        if Master == None and Var == None:
            message = "Both Parameters_Master or Parameters_Var can't be None"
            self.Exit(VLF.ErrorMessage(message))

        # ======================================================================
        # If string, import files
        if type(Master) == str:
            Master = self.ImportParameters(Master, ParameterArgs)
        if type(Var) == str:
            Var = self.ImportParameters(Var, ParameterArgs)
        MethodNS = []
        # extract namespaces for each method
        for method in self.Methods:
            MethodNS.append(self.method_config[method]["Namespace"])
        # ======================================================================
        # Perform checks of master and var and assign to self
        self.Parameters_Master = self._CheckParams(
            Master, "Parameters_Master", MethodNS
        )
        self.Parameters_Var = self._CheckParams(Var, "Parameters_Var", MethodNS)

    def _CheckParams(self, input, input_name, VLTypes):
        """Perform checks on the input file and return namespace whose attributes
        are the VLTypes"""
        # if the input is None return empty namespace
        NS = Namespace()
        if input is None:
            return NS

        # check some of the VLTypes are defined in the input
        if not set(input.__dict__).intersection(VLTypes):
            message = "{} contains none of the attrbutes {}".format(input_name, VLTypes)
            self.Exit(VLF.ErrorMessage(message))

        for nm in VLTypes:
            attr_ns = getattr(input, nm, None)  # get attribute nm from input

            # ignore if nm not an attribute
            if attr_ns is None:
                continue

            # give warning about it not being a module/namespace
            if type(attr_ns) not in (Namespace, ModuleType):
                message = "{} has attribute '{}' but it not a module or namespace.\nThis may lead to unexpected results".format(
                    input_name, nm
                )
                print(VLF.WarningMessage(message))
                continue

            # check it has a name associated with it
            if not hasattr(attr_ns, "Name"):
                message = "'{}' does not have the attribute 'Name' in {}".format(
                    nm, input_name
                )
                self.Exit(VLF.ErrorMessage(message))

            setattr(NS, nm, attr_ns)  # add the info to the namespace

        return NS

    def _CreateParameters(self, method_name):
        """
        Create parameter dictionary of attribute method_name using Parameters_Master and Var.
        """
        # ======================================================================
        # Get method_name from Parameters_Master and _Parameters_Var (if they are defined)
        Master = getattr(self.Parameters_Master, method_name, None)
        Var = getattr(self.Parameters_Var, method_name, None)

        # Check method_name is an appropriate type
        if type(Master) not in (type(None), type(Namespace())):
            print(
                VLF.WarningMessage(
                    "Variable '{}' named in Master but is not a namespace. This may lead yo unexpected results".format(
                        method_name
                    )
                )
            )
        if type(Var) not in (type(None), type(Namespace())):
            print(
                VLF.WarningMessage(
                    "Variable '{}' named in Var but is not a namespace. This may lead yo unexpected results".format(
                        method_name
                    )
                )
            )

        # ======================================================================
        # method_name isn't in Master of Var
        if Master == None and Var == None:
            return {}

        # ======================================================================
        # method_name is in Master but not in Var
        elif Var == None:
            # Check if VLFunctions.Parameters_Var function has been used to create
            # an iterator to vary parameters within master file.
            typelist = [type(val) for val in Master.__dict__.values()]
            if type(iter([])) in typelist:
                # itertor found so we consider this as a varying parameter
                Var = Namespace()
                for key, val in Master.__dict__.items():
                    if type(val) == type(iter([])):
                        setattr(Var, key, list(val))
                # Check that Name is also an iterator
                if not hasattr(Var, "Name"):
                    message = "{}.Name is not an iterable".format(method_name)
                    self.Exit(VLF.ErrorMessage(message))
                # Assign Var to class. Behaviour is the same as if _Parameters_Var
                # file had been used.
                setattr(self.Parameters_Var, method_name, Var)
            else:
                # No iterator, just a single study
                return {Master.Name: Master.__dict__}

        # ======================================================================
        # method_name is in Var

        # Check all entires in Parameters_Var have the same length
        NbNames = len(Var.Name)
        VarNames, NewVals, errVar = [], [], []
        for VariableName, NewValues in Var.__dict__.items():
            VarNames.append(VariableName)
            NewVals.append(NewValues)
            if len(NewValues) != NbNames:
                errVar.append(VariableName)

        if errVar:
            attrstr = "\n".join(["{}.{}".format(method_name, i) for i in errVar])
            message = (
                "The following attribute(s) have a different number of entries to {0}.Name in Parameters_Var:\n"
                "{1}\n\nAll attributes of {0} in Parameters_Var must have the same length.".format(
                    method_name, attrstr
                )
            )
            self.Exit(VLF.ErrorMessage(message))

        # method_name is in Master and Var
        if Master != None and Var != None:
            # Check if there are attributes defined in Var which are not in Master
            dfattrs = set(Var.__dict__.keys()) - set(
                list(Master.__dict__.keys()) + ["Run"]
            )
            if dfattrs:
                attstr = "\n".join(["{}.{}".format(method_name, i) for i in dfattrs])
                message = (
                    "The following attribute(s) are specified in Parameters_Var but not in Parameters_Master:\n"
                    "{}\n\nThis may lead to unexpected results.".format(attstr)
                )
                print(VLF.WarningMessage(message))

        # ======================================================================
        # Create dictionary for each entry in Parameters_Var
        VarRun = getattr(
            Var, "Run", [True] * NbNames
        )  # create True list if Run not an attribute of method_name
        ParaDict = {}
        for Name, NewValues, Run in zip(Var.Name, zip(*NewVals), VarRun):
            if not Run:
                continue
            base = {} if Master == None else copy.deepcopy(Master.__dict__)
            for VariableName, NewValue in zip(VarNames, NewValues):
                base[VariableName] = NewValue
            ParaDict[Name] = base

        return ParaDict

    def _get_Num_Runs(self, flags, Namespaces):
        """
        Function to get the number of runs defined in Params_master/Var for each Namespace.
        this is used to help calculate how many containers to spawn for parallel runs.
        Inputs:
        flags - list of bool's  for namespaces to run
        Namespaces -  list of namespaces
        returns:
        dict of number of runs defined for each namespace or 0 for each that Runbools is set for.
        """
        num_runs = {}
        flags = list(flags.items())
        for I, module in enumerate(Namespaces):
            # special case for CIL since it shares the GVXR namespace
            VLNamespace = self.method_config[module]["Namespace"]
            TMPDict = self._CreateParameters(VLNamespace)
            # if Run is False or Dict is empty add 0 to list 0 instead.
            if not (flags[I][1] and TMPDict):
                num_runs[module] = 0
            else:
                num_runs[module] = len(TMPDict)
        return num_runs

    def _Spread_over_Containers(self):
        """
        Function to generate a dict with a key for each VitualLab module.
        The values of the Dict are tuples the first index designates the
        container and the second designates which runs are assigned to
        that container.
        """
        from itertools import cycle

        containers = {}

        container_ids = [*range(1, self._Max_Containers + 1)]
        y = cycle(container_ids)
        for module in self.Num_runs.keys():
            runs = list(range(0, self.Num_runs[module]))
            temp = []
            for i in container_ids:
                tmp = []
                for j in runs:
                    x = next(y)
                    if x == i:
                        tmp.append(j)
                if tmp:
                    temp.append((i, tmp))
                y = cycle(container_ids)
            if temp:
                containers[module] = temp
        return containers

    def Logger(self, Text="", **kwargs):
        Prnt = kwargs.get("Print", False)

        if not hasattr(self, "LogFile"):
            print(Text)
            if self.mode in ("Interactive", "Terminal"):
                self.LogFile = None
            else:
                self.LogFile = "{}/.log/{}.log".format(self.PROJECT_DIR, self._time)
                os.makedirs(os.path.dirname(self.LogFile), exist_ok=True)
                with open(self.LogFile, "w") as f:
                    f.write(Text)
                # print("Detailed output written to {}".format(self.LogFile))
            return

        if self.mode in ("Interactive", "Terminal"):
            print(Text, flush=True)
        else:
            if Prnt:
                print(Text, flush=True)
            with open(str(self.LogFile), "a") as f:
                f.write(Text + "\n")

    def Exit(self, mess="", Cleanup=True):
        self._SetCleanup(Cleanup=Cleanup)
        sys.exit(mess)

    def _Cleanup(self, Cleanup=True):
        from . import Analytics

        # Running with base virtualLab so Report overview of VirtualLab usage
        if hasattr(self, "_Analytics") and VLconfig.VL_ANALYTICS == "True":
            Category = "{}_Overview".format(self.Simulation)
            list_AL = [
                "{}={}".format(key, value) for key, value in self._Analytics.items()
            ]
            Action = "_".join(list_AL)
            Analytics.Run(Category, Action, self._ID)

        exitstr = (
            "\n#############################\n"
            "### VirtualLab Terminated ###\n"
            "#############################\n"
        )
        if not Cleanup:
            exitstr = (
                "The temp directory {} has not been deleted.\n".format(self.TEMP_DIR)
                + exitstr
            )
        elif os.path.isdir(self.TEMP_DIR):
            shutil.rmtree(self.TEMP_DIR)
        if hasattr(self, "tcp_sock"):
            import socket

            if self._debug:
                print("closing tcp connection")
            self.tcp_sock.shutdown(socket.SHUT_RDWR)
            self.tcp_sock.close()
        print(exitstr)

    def Cleanup(self, KeepDirs=[]):
        print("Cleanup() is depreciated. You can remove this from your script")

    def do_Analytics(VL, vltype):
        """
        Function to analyse usage of VirtualLab to evidence impact for
        use in future research grant applications. Can be turned off via
        VLconfig.py. See Scripts/Common/Analytics.py for more details.
        """
        import os
        import VLconfig

        N = VL._NbJobs
        if VLconfig.VL_ANALYTICS == "true" and vltype != "Test":
            print(
                "~~~~~~~~~~~~~~~~~~~~~~\n"
                "Sending Analytics data\n"
                "~~~~~~~~~~~~~~~~~~~~~~"
            )
            from Scripts.Common import Analytics

            # Create dictionary, if one isn't defined already
            if not hasattr(VL, "_Analytics"):
                VL._Analytics = {}

            Category = "{}_{}".format(VL.Simulation, vltype)
            Action = "NJob={}_NCore={}_NNode=1_NContainers={}".format(
                vltype, N, len(VL.container_list[vltype])
            )
            # ======================================================================
            # Send information about current job
            Analytics.Run(Category, Action, VL._ID)

            # ======================================================================
            # Update Analytics dictionary with new information
            # Add information to dictionary
            if vltype not in VL._Analytics:
                VL._Analytics[vltype] = VL.Num_runs[vltype]
            else:
                VL._Analytics[vltype] += VL.Num_runs[vltype]
            return
        else:
            return

    def _git(self):
        version, branch = "<version>", "<branch>"
        try:
            from git import Repo

            repo = Repo(VLconfig.VL_DIR)
            sha = repo.head.commit.hexsha
            version = repo.git.rev_parse(sha, short=7)
            branch = repo.active_branch.name
        except:
            pass
        return "{}_{}".format(version, branch)
