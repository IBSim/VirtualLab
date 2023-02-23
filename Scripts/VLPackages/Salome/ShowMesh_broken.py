#!/usr/bin/env python

import sys
import os
import salome
import SalomePyQt
salome.salome_init()
import  SMESH, SALOMEDS
from salome.smesh import smeshBuilder
import SalomeFunc

'''
Original ShowMesh file, however mesh plugin doesn't work in container currently. ShowMesh.py opens the mesh in ParaViS instead currently.

'''

Meshes = SalomeFunc.GetArgs()

smesh = smeshBuilder.New()

MeshDict = {}
for Name, Path in Meshes.items():
    (lstMesh, status) = smesh.CreateMeshesFromMED(Path)
    for M in lstMesh:
        if len(lstMesh)==1: nm=Name
        else: nm = "{}_{}".format(Name,M.GetName())
        M.SetName(nm)
        MeshDict[nm] = M

sg = SalomePyQt.SalomePyQt()
sg.activateModule("Mesh") # Activate mesh module
sg.getObjectBrowser().expandToDepth(1) #  expand Mesh objects
# Get a ObjectID to show
smeshComp = salome.myStudy.FindComponent('SMESH') # search for component SMESH in active salome study
ID = '{}:{}'.format(smeshComp.GetID(),smeshComp.GetLastChildTag())
# Display and fit to view
salome.sg.Display(ID)
salome.sg.FitAll()
