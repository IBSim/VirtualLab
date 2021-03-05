import sys
import os
sys.dont_write_bytecode=True
import SalomeFunc
import salome
salome.salome_init()

# This function gives the ArgDict dictionary we passed to SalomeRun

ArgDict = SalomeFunc.devGetArgs()

# Import the Create function which is used to generate the mesh using the mesh parameters
Parameters = __import__(ArgDict['Name'])
Create = __import__(Parameters.File).Create

MeshRn = Create(Parameters)

if type(MeshRn)==salome.smesh.smeshBuilder.Mesh:
    isDone = MeshRn.Compute()
    SalomeFunc.MeshExport(MeshRn, ArgDict['MESH_FILE'])
elif type(MeshRn) == int:
    with open(ArgDict['RCfile'], 'w') as f:
    	f.write(str(MeshRn))

# salome.myStudy.Clear()
# salome.salome_close()
