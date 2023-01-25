import os
import sys

sys.dont_write_bytecode = True
from types import SimpleNamespace as Namespace

from Scripts.Common.VLPackages.Salome import Salome
import Scripts.Common.VLFunctions as VLF
from Scripts.Common.VLParallel import VLPool
from Scripts.Common.utils import Method_base
from Scripts.Common.VLContainer import Container_Utils as Utils


class Method(Method_base):
    def __init__(self, VL):
        super().__init__(VL)  # rune __init__ of Mthod_base
        # Add MESH_DIR to VL here as it's used by other methods (Sim,Vox)
        VL.MESH_DIR = "{}/Meshes".format(VL.PROJECT_DIR)

    def Setup(self, VL, MeshDicts, Import=False):
        # if either MeshDicts is empty or RunMesh is False we will return
        if not (self.RunFlag and MeshDicts):
            return
        sys.path.insert(0, VL.SIM_MESH)

        FileDict = {}  # Something we want to keep track of
        for MeshName, ParaDict in MeshDicts.items():
            MeshPath = "{}/{}".format(VL.MESH_DIR, MeshName)
            if Import:
                VLF.ImportUpdate("{}.py".format(MeshPath), ParaDict)

            Parameters = Namespace(**ParaDict)

            # ======================================================================
            # get file path & perform checks
            # default name is Single
            file_name, func_name = VLF.FileFuncSplit(ParaDict["File"], "Create")

            if (file_name, func_name) not in FileDict:
                # Check file in directory & get path
                FilePath = VLF.GetFilePath(
                    [VL.SIM_MESH], file_name, file_ext="py", exit_on_error=True
                )
                # Check function func_name is in the file
                a = VLF.GetFunction(FilePath, func_name, exit_on_error=True)
                File_func = [FilePath, func_name]
                FileDict[(file_name, func_name)] = File_func
            else:
                File_func = FileDict[(file_name, func_name)]

            # ======================================================================
            # Verify mesh parameters
            Verify = VLF.GetFunc(FilePath, "Verify")
            if Verify != None:
                error, warning = Verify(Parameters)
                if warning:
                    print(
                        VLF.WarningMessage(
                            "Issue with mesh {}." "\n\n{}".format(MeshName, warning)
                        )
                    )
                if error:
                    VL.Exit(
                        VLF.ErrorMessage(
                            "Issue with mesh {}." "\n\n{}".format(MeshName, error)
                        )
                    )

            # ======================================================================
            # Create dictionary for each analysis
            MeshDict = {
                "MESH_FILE": "{}.med".format(MeshPath),
                "TMP_CALC_DIR": "{}/Mesh/{}".format(VL.TEMP_DIR, MeshName),
                "FileInfo": File_func,
                "Parameters": Parameters,
            }
            if VL.mode in ("Headless", "Continuous"):
                MeshDict["LogFile"] = "{}/{}.log".format(VL.MESH_DIR, MeshName)
            else:
                MeshDict["LogFile"] = None

            os.makedirs(MeshDict["TMP_CALC_DIR"], exist_ok=True)

            self.Data[MeshName] = MeshDict.copy()

    def Spawn(self, VL, **kwargs):
        MethodName = "Mesh"
        ContainerName = "Salome"
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
            VL.Exit("Error Occurred with Mesh")
        return

    @staticmethod
    def PoolRun(VL, MeshDict, GUI=False):
        # Create directory for meshes.
        # This method supports meshes nested in sub-directories
        Meshfname = os.path.splitext(MeshDict["MESH_FILE"])[0]
        os.makedirs(os.path.dirname(Meshfname), exist_ok=True)

        # Write Parameters used to make the mesh to the mesh directory
        VLF.WriteData("{}.py".format(Meshfname), MeshDict["Parameters"])

        script = "{}/MeshRun.py".format(Salome.Dir)
        err = Salome.Run(
            script,
            DataDict=MeshDict,
            AddPath=[VL.SIM_SCRIPTS, VL.SIM_MESH],
            tempdir=MeshDict["TMP_CALC_DIR"],
            GUI=GUI,
        )
        if err:
            return "Error in Salome run"

    @VLF.kwarg_update
    def Run(self, VL, MeshCheck=None, ShowMesh=False):
        # ===========================================================================
        # MeshCheck allows you to mesh in the GUI (for debugging).Currently only 1
        # mesh can be debugged at a time. VirtualLab terminates when GUI is closed.

        if MeshCheck:
            if MeshCheck == True:
                MeshNames = list(self.Data.keys())
            elif type(MeshCheck) == str:
                MeshNames = [MeshCheck]
            elif type(MeshCheck) in (list, tuple):
                MeshNames = MeshCheck

            VL.Logger(
                "~~~~~~~~~~~~~~~~~~~~~~~~~\n"
                "~~~ Meshing using GUI ~~~\n"
                "~~~~~~~~~~~~~~~~~~~~~~~~~\n".format(Print=True)
            )
            MeshDicts = []
            for _mesh in MeshNames:
                if _mesh not in self.Data:
                    # check mesh name
                    VL.Exit(
                        VLF.ErrorMessage(
                            "'{}' specified in MeshCheck is not being created".format(
                                _mesh
                            )
                        )
                    )
                # append to list
                MeshDicts.append(self.Data[_mesh])

            AddArgs = [[True]] * len(MeshNames)  # gui = true flags

            Errorfnc = VLPool(VL, PoolRun, MeshDicts, args_list=AddArgs)
            VL.Exit(
                "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
                "~ Terminating after MeshCheck ~\n"
                "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
            )

        # ==========================================================================
        # Run Mesh routine

        VL.Logger(
            "~~~~~~~~~~~~~~~~~~~~~~~~\n"
            "~~~ Starting Meshing ~~~\n"
            "~~~~~~~~~~~~~~~~~~~~~~~~\n",
            Print=True,
        )

        Errorfnc = VLPool(VL, self.GetPoolRun(), self.Data)
        if Errorfnc:
            VL.Exit(
                VLF.ErrorMessage(
                    "\nThe following meshes finished with errors:\n{}".format(Errorfnc)
                ),
                Cleanup=False,
            )

        VL.Logger(
            "~~~~~~~~~~~~~~~~~~~~~~~~\n"
            "~~~ Meshing Complete ~~~\n"
            "~~~~~~~~~~~~~~~~~~~~~~~~\n",
            Print=True,
        )

        # ==========================================================================
        # Open meshes in GUI to view

        if ShowMesh:
            VL.Logger("\n### Opening mesh files in Salome ###\n", Print=True)
            ArgDict = {
                name: "{}/{}.med".format(VL.MESH_DIR, name) for name in self.Data.keys()
            }
            Script = "{}/ShowMesh.py".format(Salome.Dir)
            Salome.Run(Script, DataDict=ArgDict, tempdir=VL.TEMP_DIR, GUI=True)
            VL.Exit("\n### Terminating after mesh viewing ###")
