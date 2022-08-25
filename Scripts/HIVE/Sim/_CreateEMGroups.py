# SALOME python script

import sys
import numpy as np
import  SMESH, SALOMEDS
from salome.smesh import smeshBuilder
smesh = smeshBuilder.New()
from SalomeFunc import GetArgs


ArgDict = GetArgs()
(Meshes, status) = smesh.CreateMeshesFromMED(ArgDict["MeshFile"])
Sample = Meshes[0]

EM_Groups = ArgDict['EM_Groups']

Elements = []
for a in EM_Groups:
	Elements.extend(a.tolist())

grp = Sample.CreateEmptyGroup(SMESH.VOLUME, "_EMgrp")
grp.Add(Elements)

for i, El in enumerate(EM_Groups):
	grp = Sample.CreateEmptyGroup(SMESH.VOLUME, "_{}".format(i))
	grp.Add(El.tolist())

Sample.ExportMED( ArgDict["tmpMesh"], auto_groups=0, minor=40, overwrite=1,meshPart=None,autoDimension=1)
