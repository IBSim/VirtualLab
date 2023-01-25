import os
import sys

sys.dont_write_bytecode = True
from types import SimpleNamespace as Namespace

from Scripts.Common.VLParallel import VLPool
from Scripts.Common.utils import Method_base
from Scripts.Common.VLContainer import Container_Utils as Utils

"""
Template file for creating a new method. Create a copy of this file as #MethodName.py,
which is the name of the new method, and edit as desired. Any file starting with
_ are ignored.
"""


class Method(Method_base):
    def __init__(self, VL):
        """
        Otional class initiation function. If this is used you will also need
        initiate Method_base using 'super().__init__(VL)' (see Mesh method for
        more details).
        """

    def Setup(self, VL, MethodDicts, Import=False):
        """
        Functions used for setting things up. Parameters associated with the method
        name for the Parameters_Master and Var files are passed as the
        'MethodDicts' argument.

        Information is assigned to the dictionary self.Data for use in the self.Run
        and self.PoolRun functions. See the other available methods for examples.

        If the flag for running a method, set using Run#MethodName in VirtualLab.Parameters,
        or #MethodName is not included in the parameters file(s) you will likely
        want to skip this function. Add

        'if not (self.RunFlag and MethodDicts): return'

        at the top of this file to return immediately.


        """

        if not (self.RunFlag and MethodDicts):
            return

        for MethodName, MethodParameters in MethodDicts.items():
            # Perform some checks on the info in MethodParams

            """
            Create a dictionary containing the parameters and other useful
            information for use by the PoolRun function. This information is
            assigned to self.Data.
            """
            AnalysisDict = {
                "Parameters": Namespace(**MethodParameters),
                # Add other useful info here also
            }
            self.Data[MethodName] = AnalysisDict

    @staticmethod
    def PoolRun(VL, AnalysisDict, **kwargs):
        """
        Functions which does something with the information from AnalysisDict.
        See Mesh, Sim, DA for more details.

        Note: This must have the decorator @staticmethod as it does not take the
        argument 'self'.
        """

    def Run(self, VL, **kwargs):
        """
        This is the function called when running VirtualLab.#MethodName in the
        run file with the commandline option Module=True.

        If this option if set to False (or not set at all since the default is
        False) the Function "Spawn" is called instead.

        This Function uses the information assigned to self.Data to perform
        analysis.

        Use VLPool for high throughput parallelisation. This uses the information
        from the 'Launcher' and 'NbJobs' specified in the settings to run the analyses
        in parallel.
        Note: self.GetPoolRun() is a safer way of getting the PoolRun function.

        Check for errors and exit if there are any.
        """
        VL.Logger(
            "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
            "~~~ Starting #MethodName ~~~\n"
            "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n",
            Print=True,
        )

        AnalysisDicts = list(self.Data.values())  # Data assigned during Setup

        Errorfnc = VLPool(VL, self.GetPoolRun(), AnalysisDicts)
        if Errorfnc:
            VL.Exit(
                VLF.ErrorMessage(
                    "The following #MethodName routine(s) finished with errors:\n{}".format(
                        Errorfnc
                    )
                ),
                Cleanup=False,
            )

        VL.Logger(
            "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
            "~~~ #MethodName Complete ~~~\n"
            "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n",
            Print=True,
        )

    def Spawn(self, VL, **kwargs):
        """
        This is the function called when running VirtualLab.#MethodName
        within the VL_Manger container (i.e. the RunFile).

        This Function sends a message to the host to spawn a container
        "#ContainerName". This refers to one of the Containers defined
        in VL_modules.yaml.

        ***************************************************************
        ***********  Note for using multiple containers   *************
        ***************************************************************
        VirtualLab can, with some setup, be configured to spread defined
        jobs over multiple containers.However, this can be very
        problematic and resource (particularly ram) intensive.

        Therefore, for running in parallel we recommend using pathos or mpi
        set via the VL._Launcher option (see VLParallel.py). However, if
        for whatever reason this is not an option an example of how to
        implement this can be found in GVXR.py.

        """
        MethodName = "#MethodName"
        ContainerName = "#ContainerName"
        Cont_runs = VL.container_list.get(MethodName, None)
        if Cont_runs == None:
            print(
                f"Warning: Method {MethodName} was called in the inputfile but has no coresponding"
                f" namespace in the parameters file. To remove this warning message please set Run{MethodName}=False."
            )
            return

        return_value = Utils.Spawn_Container(
            VL,
            Cont_id=1,
            Tool=ContainerName,
            Method_Name=MethodName,
            Num_Cont=1,
            Cont_runs=Cont_runs,
            Parameters_Master=VL.Parameters_Master_str,
            Parameters_Var=VL.Parameters_Var_str,
            Project=VL.Project,
            Simulation=VL.Simulation,
            Settings=VL.settings_dict,
            tcp_socket=VL.tcp_sock,
            run_args=kwargs,
        )

        if return_value != "0":
            # an error occurred so exit VirtualLab
            VL.Exit(f"Error Occurred with {MethodName}")
        return
