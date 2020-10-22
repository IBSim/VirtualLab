import sys
import os
sys.dont_write_bytecode=True
import SalomeFunc
import salome
salome.salome_init()

# This function gives the ArgDict dictionary we passed to SalomeRun
kwargs = SalomeFunc.GetArgs(sys.argv[1:])

# Import the Create function which is used to generate the mesh using the mesh parameters
Parameters = __import__(kwargs['Parameters'])
Create = __import__(Parameters.File).Create

MeshRn = Create(Parameters)

if kwargs.get('ConfigFile'):
    import config
    config.MeshStore(MeshRn, kwargs['MESH_FILE'], kwargs['RCfile'], Parameters=Parameters)
else :
    SalomeFunc.MeshStore(MeshRn, kwargs['MESH_FILE'], kwargs['RCfile'])

salome.myStudy.Clear()
salome.salome_close()
