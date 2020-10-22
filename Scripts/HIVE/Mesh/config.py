import sys
import os
sys.dont_write_bytecode=True
import salome
from SalomeFunc import MeshExport

def MeshStore(MeshRn,MeshFile,RCfile,**kwargs):
    Parameters = kwargs["Parameters"]
    if type(MeshRn)==salome.smesh.smeshBuilder.Mesh:
        if hasattr(Parameters,'CoilType'):
            print('Using sample mesh information to create mesh for ERMES')
            from EM.EMChamber import CreateEMMesh
            Meshes = CreateEMMesh(MeshRn, Parameters)
            if Meshes==False:
                pass
            else :
                SampleMesh, ERMESMesh = Meshes[0:2]
                # smesh.SetName(SampleMesh, 'Sample')
                # smesh.SetName(ERMESMesh, 'xERMES')
                MeshExport(SampleMesh,MeshFile)
                MeshExport(ERMESMesh,MeshFile, Overwrite = 0)
        else:
            MeshRn.Compute()
            MeshExport(MeshRn,MeshFile)

    elif type(MeshRn)==int:
        print(MeshRn)
