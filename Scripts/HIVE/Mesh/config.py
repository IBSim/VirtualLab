import sys
import os
sys.dont_write_bytecode=True
import salome
import SalomeFunc

def MeshStore(MeshRn,MeshFile,RCfile,**kwargs):
    Parameters = kwargs["Parameters"]
    if type(MeshRn)==salome.smesh.smeshBuilder.Mesh:
        if hasattr(Parameters,'CoilType'):
            print('Using sample mesh information to create mesh for ERMES')
            from EM.EMChamber import CreateEMMesh
            Meshes = CreateEMMesh(MeshRn, Parameters)
            if type(Meshes)==int:
                SalomeFunc.MeshRC(RCfile, Meshes)
            else :
                SampleMesh, ERMESMesh = Meshes[0:2]
                # smesh.SetName(SampleMesh, 'Sample')
                # smesh.SetName(ERMESMesh, 'xERMES')
                SalomeFunc.MeshExport(SampleMesh,MeshFile)
                SalomeFunc.MeshExport(ERMESMesh,MeshFile, Overwrite = 0)
        else:
            MeshRn.Compute()
            SalomeFunc.MeshExport(MeshRn,MeshFile)

    elif type(MeshRn)==int:
        SalomeFunc.MeshRC(RCfile, MeshRn)
