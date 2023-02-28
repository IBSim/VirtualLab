import sys
import os
sys.dont_write_bytecode=True
import SalomeFunc
import salome
from salome.geom import geomBuilder
salome.salome_init()

# This function gives the ArgDict dictionary we passed to SalomeRun
MeshDict = SalomeFunc.GetArgs()
Parameters = MeshDict['Parameters']

# Import the function which is used to generate the mesh using the mesh parameters
Create = SalomeFunc.GetFunc(*MeshDict['FileInfo'])
MeshRn = Create(Parameters)

if MeshDict.get('Debug',False):
    pass
elif type(MeshRn)==salome.smesh.smeshBuilder.Mesh:
    isDone = MeshRn.Compute()
    SalomeFunc.MeshExport(MeshRn, MeshDict['MESH_FILE'])

    # If Mesh has the attribute 'Export Geom' then the geometry used to mesh will
    # be exported alognside the mesh. Available formats are: stp,stl and xao.
    # NOTE: xao exports all groups associated with the geomtry as well
    if hasattr(Parameters,'ExportGeom'):
        geompy = geomBuilder.New()
        SampleGeom = MeshRn.GetShape()
        fname = os.path.splitext(MeshDict['MESH_FILE'])[0] # Same name as mesh

        if Parameters.ExportGeom.lower() in ('step','stp'):
            import GEOM
            geompy.ExportSTEP(SampleGeom, "{}.stp".format(fname), GEOM.LU_METER )
        elif Parameters.ExportGeom.lower() == 'stl':
            geompy.ExportSTL(SampleGeom, "{}.stl".format(fname), True, 0.001, True)
        elif Parameters.ExportGeom.lower() == 'xao':
            groups = geompy.GetExistingSubObjects(SampleGeom,True)
            geompy.ExportXAO(SampleGeom, groups, [], "", "{}.xao".format(fname), "")

# salome.myStudy.Clear()
# salome.salome_close()
