#!/usr/bin/env python

###
### This file is generated automatically by SALOME v9.3.0 with dump python functionality
###

import sys
import os
import salome
import SalomePyQt
salome.salome_init()

###
### SMESH component
###

import  SMESH, SALOMEDS
from salome.smesh import smeshBuilder
smesh = smeshBuilder.New()

MeshDict = {}
MeshList = sys.argv[1:]
for Mesh in MeshList:
	Name = os.path.splitext(os.path.basename(Mesh))[0]
	(lstMesh, status) = smesh.CreateMeshesFromMED(Mesh)
	MeshDict[Name] = {M.GetName():M for M in lstMesh }

sg = SalomePyQt.SalomePyQt()
sg.activateModule("Mesh") # Activate mesh module
sg.getObjectBrowser().expandToDepth(1) #  expand Mesh objects
# Get a ObjectID to show
smeshComp = salome.myStudy.FindComponent('SMESH') # search for component SMESH in active salome study
ID = '{}:{}'.format(smeshComp.GetID(),smeshComp.GetLastChildTag())
# Display and fit to view
salome.sg.Display(ID)
salome.sg.FitAll()
