import sys
import os
sys.dont_write_bytecode=True
import SalomeFunc
import salome
salome.salome_init()
from salome.geom import geomBuilder
import GEOM

geompy = geomBuilder.New()

# This function gives the ArgDict dictionary we passed to SalomeRun
MeshDict = SalomeFunc.GetArgs()

# Import the Create function which is used to generate the mesh using the mesh parameters
Parameters = MeshDict['Parameters']
MeshFile = MeshDict['MESH_FILE']

Create = __import__(Parameters.File).Create
MeshRn = Create(Parameters)

if MeshDict.get('Debug',False):
    pass
elif type(MeshRn)==salome.smesh.smeshBuilder.Mesh:
    MeshRn.Compute()
    SalomeFunc.MeshExport(MeshRn,MeshFile)

    # Need geom object for HIVE to orientate sample with coil
    # .xao file allows you to store the geometry and its groups.
    xaofile = "{}.xao".format(os.path.splitext(MeshFile)[0])
    SampleGeom = MeshRn.GetShape()
    GrpsRequired = ['CoilFace','PipeIn','PipeOut','SampleSurface']
    GrpGeom = []
    for SubShape in geompy.GetExistingSubObjects(SampleGeom,True):
        if SubShape.GetName() in GrpsRequired:
            GrpGeom.append(SubShape)

    geompy.ExportXAO(SampleGeom, GrpGeom, [], "", xaofile, "")

    if hasattr(Parameters,'ExportGeom'):
        SampleGeom = MeshRn.GetShape()
        fname = os.path.splitext(MeshDict['MESH_FILE'])[0] # Same name as mesh

        if Parameters.ExportGeom.lower() in ('step','stp'):
            geompy.ExportSTEP(SampleGeom, "{}.stp".format(fname), GEOM.LU_METER )
        elif Parameters.ExportGeom.lower() == 'stl':
            geompy.ExportSTL(SampleGeom, "{}.stl".format(fname), True, 0.001, True)
        elif Parameters.ExportGeom.lower() == 'xao':
            groups = geompy.GetExistingSubObjects(SampleGeom,True)
            geompy.ExportXAO(SampleGeom, groups, [], "", "{}.xao".format(fname), "")

# salome.myStudy.Clear()
# salome.salome_close()
