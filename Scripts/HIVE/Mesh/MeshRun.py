import sys
import os
sys.dont_write_bytecode=True
import SalomeFunc
import salome
salome.salome_init()

# This function gives the ArgDict dictionary we passed to SalomeRun
kwargs = SalomeFunc.GetArgs(sys.argv[1:])

# Import the Create function which is used to generate the mesh using the mesh parameters
Parameters = __import__(kwargs['Name'])
MeshFile = kwargs['MESH_FILE']

Create = __import__(Parameters.File).Create
MeshRn = Create(Parameters)

if type(MeshRn)==salome.smesh.smeshBuilder.Mesh:
    if hasattr(Parameters,'CoilType'):
        print('Using sample mesh information to create mesh for ERMES')
        from EM.EMChamber import CreateEMMesh
        Meshes = CreateEMMesh(MeshRn, Parameters)
        if type(Meshes)==int:
            with open(kwargs['RCfile'],'w') as f:
            	f.write(str(Meshes))
        else :
            SampleMesh, ERMESMesh = Meshes[0:2]
            # smesh.SetName(SampleMesh, 'Sample')
            # smesh.SetName(ERMESMesh, 'xERMES')
            SalomeFunc.MeshExport(SampleMesh,MeshFile)
            SalomeFunc.MeshExport(ERMESMesh,MeshFile, Overwrite = 0)
    else:
        MeshRn.Compute()
        SalomeFunc.MeshExport(MeshRn,MeshFile)
        # only called by dev scripts
        if 'STEP' in kwargs:
            from salome.geom import geomBuilder
            import GEOM
            geompy = geomBuilder.New()

            # xao file allows you to store the geometry and its groups.
            # Need some groups to orientate the sample with the Coil
            xaofile = "{}.xao".format(os.path.splitext(MeshFile)[0])
            SampleGeom = MeshRn.GetShape()
            GrpsRequired = ['CoilFace','PipeIn','PipeOut','SampleSurface']
            GrpGeom = []
            for SubShape in geompy.GetExistingSubObjects(SampleGeom,True):
                if SubShape.GetName() in GrpsRequired:
                    GrpGeom.append(SubShape)

            geompy.ExportXAO(SampleGeom, GrpGeom, [], "", xaofile, "")

elif type(MeshRn)==int:
    with open(kwargs['RCfile'],'w') as f:
    	f.write(str(MeshRn))

# salome.myStudy.Clear()
# salome.salome_close()
