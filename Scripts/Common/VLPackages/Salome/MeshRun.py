import sys
import os
sys.dont_write_bytecode=True
import SalomeFunc
import salome
salome.salome_init()

# This function gives the ArgDict dictionary we passed to SalomeRun

MeshDict = SalomeFunc.GetArgs()

# Import the Create function which is used to generate the mesh using the mesh parameters
Parameters = MeshDict['Parameters']
Create = __import__(Parameters.File).Create

MeshRn = Create(Parameters)

if type(MeshRn)==salome.smesh.smeshBuilder.Mesh:
    isDone = MeshRn.Compute()
    SalomeFunc.MeshExport(MeshRn, MeshDict['MESH_FILE'])

# salome.myStudy.Clear()
# salome.salome_close()
