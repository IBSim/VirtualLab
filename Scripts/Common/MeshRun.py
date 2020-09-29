from subprocess import Popen
import sys
import os
sys.dont_write_bytecode=True
from SalomeFunc import GetArgs
import salome

# This function gives the ArgDict dictionary we passed to SalomeRun
kwargs = GetArgs(sys.argv[1:])

Parameters = __import__(kwargs['Parameters'])
Create = __import__(Parameters.File).Create

RC = Create(Parameter = Parameters, MeshFile = kwargs['MESH_FILE'])

salome.myStudy.Clear()
salome.salome_close()

if RC:
    print('')
    sys.exit(RC)
