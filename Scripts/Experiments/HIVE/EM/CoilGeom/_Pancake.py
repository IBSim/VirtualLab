#!/usr/bin/env python

###
### This file is generated automatically by SALOME v9.3.0 with dump python functionality
###

import sys
import salome

salome.salome_init()
import salome_notebook
notebook = salome_notebook.NoteBook()
sys.path.insert(0, r'/home/rhydian/Documents/Scripts/Simulation/VirtualLab/Scripts/HIVE/EM')

###
### GEOM component
###

import GEOM
from salome.geom import geomBuilder
import math
import SALOMEDS


geompy = geomBuilder.New()

O = geompy.MakeVertex(0, 0, 0)
OX = geompy.MakeVectorDXDYDZ(1, 0, 0)
OY = geompy.MakeVectorDXDYDZ(0, 1, 0)
OZ = geompy.MakeVectorDXDYDZ(0, 0, 1)
Round_Flat_Pancake_Induction_Coil_Solutions_stp_1 = geompy.ImportSTEP("/home/rhydian/Downloads/Round Flat Pancake Induction Coil Solutions.stp", False, True)
HIVE_coil_stp_1 = geompy.ImportSTEP("/home/rhydian/Documents/Scripts/Simulation/VirtualLab/Scripts/HIVE/EM/HIVE_coil.stp", False, True)
Rotation_1 = geompy.MakeRotation(Round_Flat_Pancake_Induction_Coil_Solutions_stp_1, OY, 18*math.pi/180.0)
Translation_1 = geompy.MakeTranslation(Rotation_1, 0.047, 0.048, -0.065)
Rotation_2 = geompy.MakeRotation(Translation_1, OY, 185*math.pi/180.0)
Rotation_3 = geompy.MakeRotation(Rotation_2, OZ, -55*math.pi/180.0)
(imported, Sample, [], [CoilFace, PipeIn, PipeOut, SampleSurface], []) = geompy.ImportXAO("/home/rhydian/Documents/Scripts/Simulation/VLResults/HIVE/HIVE_Component/Meshes/SampleHIVE.xao")
geompy.ExportSTEP(Rotation_3, "/home/rhydian/Documents/Scripts/Simulation/VirtualLab/Scripts/HIVE/EM/Pancake_coil.step", GEOM.LU_METER )
geompy.addToStudy( O, 'O' )
geompy.addToStudy( OX, 'OX' )
geompy.addToStudy( OY, 'OY' )
geompy.addToStudy( OZ, 'OZ' )
geompy.addToStudy( Round_Flat_Pancake_Induction_Coil_Solutions_stp_1, 'Round Flat Pancake Induction Coil Solutions.stp_1' )
geompy.addToStudy( HIVE_coil_stp_1, 'HIVE_coil.stp_1' )
geompy.addToStudy( Rotation_1, 'Rotation_1' )
geompy.addToStudy( Translation_1, 'Translation_1' )
geompy.addToStudy( Rotation_2, 'Rotation_2' )
geompy.addToStudy( Rotation_3, 'Rotation_3' )
geompy.addToStudy( Sample, 'Sample' )
geompy.addToStudyInFather( Sample, CoilFace, 'CoilFace' )
geompy.addToStudyInFather( Sample, PipeIn, 'PipeIn' )
geompy.addToStudyInFather( Sample, PipeOut, 'PipeOut' )
geompy.addToStudyInFather( Sample, SampleSurface, 'SampleSurface' )

###
### PARAVIS component
###

import pvsimple
pvsimple.ShowParaviewView()
# trace generated using paraview version 5.6.0-RC1

#### import the simple module from the paraview
from pvsimple import *
#### disable automatic camera reset on 'Show'
pvsimple._DisableFirstRenderCameraReset()

#### saving camera placements for all active views




if salome.sg.hasDesktop():
  salome.sg.updateObjBrowser()
