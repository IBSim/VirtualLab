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
    def Setup(self, VL, CILdicts, Import=False):
        """setup CT reconstruction with CIL"""
        # if RunCIL is False or CILdicts is empty dont perform Simulation and return instead.
        if not (self.RunFlag and CILdicts):
            return

        self.Data = {}
        for CILName, CILParams in CILdicts.items():
            Parameters = Namespace(**CILParams)

            CILdict = {
                "work_dir": "{}/GVXR-Images".format(VL.PROJECT_DIR),
                "Name": CILName,
            }
            # Define flag to display visualisations
            if VL.mode == "Headless":
                CILdict["Headless"] = True
            else:
                CILdict["Headless"] = False

            if hasattr(Parameters, "Nikon_file"):
                CILdict["Nikon"] = Parameters.Nikon_file
            else:
                CILdict["Nikon"] = None

            # if hasattr(Parameters,'Beam_Pos_units'):
            #    CILdict['Beam_Pos_units'] = Parameters.Beam_Pos_units
            # else:
            #    CILdict['Beam_Pos_units'] = 'm'

            CILdict["Beam"] = [
                Parameters.Beam_PosX,
                Parameters.Beam_PosY,
                Parameters.Beam_PosZ,
            ]

            # if hasattr(Parameters,'Detect_Pos_units'):
            #    CILdict['Det_Pos_units'] = Parameters.Detect_Pos_units
            # else:
            #    CILdict['Det_Pos_units'] = 'm'

            if hasattr(Parameters, "Spacing_X"):
                CILdict["Spacing_X"] = Parameters.Spacing_X
            else:
                CILdict["Spacing_X"] = 0.5

            if hasattr(Parameters, "Spacing_Y"):
                CILdict["Spacing_Y"] = Parameters.Spacing_Y
            else:
                CILdict["Spacing_Y"] = 0.5

            CILdict["Detector"] = [
                Parameters.Detect_PosX,
                Parameters.Detect_PosY,
                Parameters.Detect_PosZ,
            ]

            CILdict["Model"] = [
                Parameters.Model_PosX,
                Parameters.Model_PosY,
                Parameters.Model_PosZ,
            ]

            CILdict["Pix_X"] = Parameters.Pix_X

            CILdict["Pix_Y"] = Parameters.Pix_Y

            # if hasattr(Parameters,'Model_Pos_units'):
            #    CILdict['Model_Pos_units'] = Parameters.Model_Pos_units
            # else:
            #    CILdict['Model_Pos_units'] = 'm'

            if hasattr(Parameters, "rotation"):
                CILdict["rotation"] = Parameters.rotation

            if hasattr(Parameters, "num_projections"):
                CILdict["num_projections"] = Parameters.num_projections

            if hasattr(Parameters, "angular_step"):
                CILdict["angular_step"] = Parameters.angular_step

            if hasattr(Parameters, "image_format"):
                CILdict["im_format"] = Parameters.image_format

            self.Data[CILName] = CILdict.copy()
        return

    # *******************************************************************

    @staticmethod
    def PoolRun(VL, AnalysisDict, **kwargs):
        """
        Functions which does something with the information from AnalysisDict.
        See Mesh, Sim, DA for more details.

        Note: This must have the decorator @staticmethod as it does not take the
        argument 'self'.

        """
        from Scripts.Common.VLPackages.CIL.CT_reconstruction import CT_Recon
        Errorfnc = CT_Recon(**AnalysisDict)
        if Errorfnc:
            return Errorfnc

    def Run(self, VL, **kwargs):
        """
        This is the function called when running VirtualLab.#MethodName in the
        run file with the command line option Module=True.

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
        if not self.Data:
            return
        ####################################
        ## Test for CIL install ########
        try:
            from cil.framework import AcquisitionGeometry

            VL.Logger("Success 'CIL' is installed", print=True)
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "module CIL is not installed are you sure "
                "you are running in the correct container?"
            )
        #########################################
        from Scripts.Common.VLPackages.CIL.CT_reconstruction import CT_Recon

        VL.Logger('\n### Starting CIL ###\n', Print=True)

        AnalysisDicts = list(self.Data.values())  # Data assigned during Setup

        Errorfnc = VLPool(VL, self.GetPoolRun(), AnalysisDicts)
        if Errorfnc:
            VL.Exit(
                VLF.ErrorMessage(
                    "The following CIL routine(s) finished with errors:\n{}".format(
                        Errorfnc
                    )
                ),
                Cleanup=False,
            )

        VL.Logger("\n### CIL Complete ###", Print=True)

    def Spawn(self, VL, **kwargs):
        """
        This is the function calling the CIL method from VLSetup

        If CIL is called by VLModule "Run" is called instead.

        This Function sends a message to the host to spawn a CIL container.
        Which is one of the Containers defined in VL_modules.yaml.

        """
        MethodName = "CT_Recon"
        ContainerName = "CIL"
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
            Num_Cont=len(VL.container_list[MethodName]),
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
            VL.Exit("Error Occurred with CIL")
        return
