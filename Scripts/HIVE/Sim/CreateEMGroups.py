# SALOME python script
import time
import sys
import numpy as np
import  SMESH, SALOMEDS
from salome.smesh import smeshBuilder
smesh = smeshBuilder.New()
from SalomeFunc import GetArgs


ArgDict = GetArgs(sys.argv[1:])
(Meshes, status) = smesh.CreateMeshesFromMED(ArgDict["MeshFile"])
EMdata = np.load(ArgDict["EMLoadFile"])
Sample = Meshes[0]

Elements = list(map(int,EMdata[:,0]))
grp = Sample.CreateEmptyGroup(SMESH.VOLUME, "EMLoadElements")
grp.Add(Elements)
st = time.time()
for El in Elements:
	grp = Sample.CreateEmptyGroup(SMESH.VOLUME, "M{}".format(El))
	grp.Add([El])
print(time.time()-st)
Sample.ExportMED( ArgDict["tmpMesh"], auto_groups=0, minor=40, overwrite=1,meshPart=None,autoDimension=1)
