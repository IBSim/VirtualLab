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

    ### dev work
    elif type(MeshRn)==tuple:
        if hasattr(Parameters,'CoilType'):
            print('Using sample mesh information to create mesh for ERMES')
            from EM.EMChamber import CreateEMMesh
            Meshes = CreateEMMesh(MeshRn[0], Parameters)
            if type(Meshes)==int:
                SalomeFunc.MeshRC(RCfile, Meshes)
            else :
                SampleMesh, ERMESMesh = Meshes[0:2]
                # smesh.SetName(SampleMesh, 'Sample')
                # smesh.SetName(ERMESMesh, 'xERMES')
                SalomeFunc.MeshExport(SampleMesh,MeshFile)
                SalomeFunc.MeshExport(ERMESMesh,MeshFile, Overwrite = 0)
        else:
            MeshRn[0].Compute()
            SalomeFunc.MeshExport(MeshRn[0],MeshFile)

            # only called by dev scripts
            if 'STEP' in kwargs:
                from salome.geom import geomBuilder
                import GEOM
                geompy = geomBuilder.New()
                xaofile = "{}.xao".format(os.path.splitext(MeshFile)[0])
                ls = MeshRn[1]
                geompy.ExportXAO(ls[0], ls[1:], [], "", xaofile, "")
