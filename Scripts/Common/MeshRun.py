import sys
import os
sys.dont_write_bytecode=True
from SalomeFunc import GetArgs
import salome
salome.salome_init()

# This function gives the ArgDict dictionary we passed to SalomeRun
kwargs = GetArgs(sys.argv[1:])

Parameters = __import__(kwargs['Parameters'])
Create = __import__(Parameters.File).Create

RC = Create(Parameter = Parameters, MeshFile = kwargs['MESH_FILE'])

# nb = salome.myStudy.NewBuilder()
# comp = salome.myStudy.FindComponent('GEOM')
# iterator = salome.myStudy.NewChildIterator( comp )
# while iterator.More():
#     sobj = iterator.Value()
#     # print(sobj.GetName())
#     sobj.UnRegister()
#     iterator.Next()

salome.myStudy.Clear()
salome.salome_close()

if RC:
    print('')
    sys.exit(RC)
