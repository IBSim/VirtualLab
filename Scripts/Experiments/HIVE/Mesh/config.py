import os

from Scripts.Common.VLPackages.Salome import Salome
import Scripts.Common.VLFunctions as VLF
from Scripts.Common.VLParallel import VLPool
from Scripts.Methods.Mesh import Method as Method_default

class Method(Method_default):
    @staticmethod
    def PoolRun(VL, MeshDict,GUI=False):
        # Create directory for meshes.
        # This method supports meshes nested in sub-directories
        Meshfname = os.path.splitext(MeshDict['MESH_FILE'])[0]
        os.makedirs(os.path.dirname(Meshfname),exist_ok=True)

        # Write Parameters used to make the mesh to the mesh directory
        VLF.WriteData("{}.py".format(Meshfname), MeshDict['Parameters'])

        # Use a user-made MeshRun file instead.
        script = '{}/MeshRun.py'.format(VL.SIM_MESH)
        err = Salome.Run(script, DataDict = MeshDict, AddPath=[VL.SIM_SCRIPTS,VL.SIM_MESH],
                         tempdir=MeshDict['TMP_CALC_DIR'],GUI=GUI)
        if err:
            return "Error in Salome run"
