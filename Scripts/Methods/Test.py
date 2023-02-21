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
    def Setup(self, VL, TestDicts, Import=False):
        """
        Setup for Tests of container communications
        """
        from types import SimpleNamespace as Namespace

        # if RunTest is False or TestDicts is empty dont perform Simulation and return instead.
        if not (self.RunFlag and TestDicts):
            return

        VL.TestData = {}

        for TestName, TestParams in TestDicts.items():
            Parameters = Namespace(**TestParams)
            TestDict = {}
            # Define flag to display visualisations
            if VL.mode == "Headless":
                TestDict["Headless"] = True
            else:
                TestDict["Headless"] = False
            #
            if hasattr(Parameters, "msg"):
                TestDict["Message"] = Parameters.msg
            else:
                raise ValueError(
                    "You must Specify a test message in the params file to display."
                )

            self.Data[TestName] = TestDict.copy()

    @staticmethod
    def PoolRun(VL, AnalysisDict, **kwargs):
        """
        Functions which does something with the information from AnalysisDict.
        See Mesh, Sim, DA for more details.

        Note: This must have the decorator @staticmethod as it does not take the
        argument 'self'.
        """
        pass

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
            "~~~ Starting Test ~~~\n"
            "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n",
            Print=True,
        )

        if not self.Data:
            return
        VL.Logger("\n### Starting Comms Test ###\n", Print=True)

        for key in self.Data.keys():
            data = self.Data[key]
            
            ContainerInfo = {'ContainerName':'Test_Comms',
                            }
            command = "bash /usr/bin/jokes.sh {}".format(data["Message"])
                            
                                            
            RC = Utils.Exec_Container(ContainerInfo, command)

        VL.Logger(
            "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
            "~~~ Test Complete ~~~\n"
            "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n",
            Print=True,
        )


