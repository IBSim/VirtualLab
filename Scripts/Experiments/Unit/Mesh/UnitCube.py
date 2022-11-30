import sys
sys.dont_write_bytecode=True
import numpy as np
import os

'''
In this script the geometry and mesh we are creating is defined in the function 'Create', with dimensional arguments and mesh arguments passed to it. The 'test' function provides dimensions for when the script is loaded manually in to Salome and not via a parametric study. The error function is imported during the setup of parametric studies to check for any geometrical errors which may arise.
'''
def Create(Parameter):
	#!/usr/bin/env python3

	from salome.geom import geomBuilder
	from salome.smesh import smeshBuilder
	import  SMESH
	import salome_version

	if salome_version.getVersions()[0] < 9:
		import salome
		theStudy = salome.myStudy
		geompy = geomBuilder.New(theStudy)
		smesh = smeshBuilder.New(theStudy)
	else :
		geompy = geomBuilder.New()
		smesh = smeshBuilder.New()

	O = geompy.MakeVertex(0, 0, 0)
	OX = geompy.MakeVectorDXDYDZ(1, 0, 0)
	OY = geompy.MakeVectorDXDYDZ(0, 1, 0)
	OZ = geompy.MakeVectorDXDYDZ(0, 0, 1)
	Box_1 = geompy.MakeBoxDXDYDZ(1, 1, 1)
	Face_1 = geompy.CreateGroup(Box_1, geompy.ShapeType["FACE"])
	geompy.UnionIDs(Face_1, [23])
	Face_2 = geompy.CreateGroup(Box_1, geompy.ShapeType["FACE"])
	geompy.UnionIDs(Face_2, [27])

	#geompy.addToStudy( O, 'O' )
	#geompy.addToStudy( OX, 'OX' )
	#geompy.addToStudy( OY, 'OY' )
	#geompy.addToStudy( OZ, 'OZ' )
	#geompy.addToStudy( Box_1, 'Box_1' )
	#geompy.addToStudyInFather( Box_1, Face_1, 'Face_1' )
	#geompy.addToStudyInFather( Box_1, Face_2, 'Face_2' )

	###
	### SMESH component
	###

	Mesh_1 = smesh.Mesh(Box_1)
	Regular_1D = Mesh_1.Segment()
	Regular_1D.NumberOfSegments(2)
	Quadrangle_2D = Mesh_1.Quadrangle(algo=smeshBuilder.QUADRANGLE)
	Hexa_3D = Mesh_1.Hexahedron(algo=smeshBuilder.Hexa)
	MFace_1 = Mesh_1.GroupOnGeom(Face_1,'Face_1',SMESH.FACE)
	MFace_2 = Mesh_1.GroupOnGeom(Face_2,'Face_2',SMESH.FACE)
	Mesh_1.Compute()

	return Mesh_1

	## Set names of Mesh objects
	#smesh.SetName(Regular_1D.GetAlgorithm(), 'Regular_1D')
	#smesh.SetName(Hexa_3D.GetAlgorithm(), 'Hexa_3D')
	#smesh.SetName(Quadrangle_2D.GetAlgorithm(), 'Quadrangle_2D')
	#smesh.SetName(Number_of_Segments_1, 'Number of Segments_1')
	#smesh.SetName(Mesh_1.GetMesh(), 'Mesh_1')
	#smesh.SetName(MFace_1, 'Face_1')
	#smesh.SetName(MFace_2, 'Face_2')


class TestDimensions():
	def __init__(self):
		### Geometric parameters
		self.MeshName = 'Test'

if __name__ == '__main__':
	if len(sys.argv) == 1:
		Create(Parameter = TestDimensions(),MeshFile = None)
	# 1 argument provided which is the parameter file
	elif len(sys.argv) == 2:
		ParameterFile = sys.argv[1]
		sys.path.insert(0, os.path.dirname(ParameterFile))
		Parameters = __import__(os.path.splitext(os.path.basename(ParameterFile))[0])
		Create(Parameter = Parameters,MeshFile = None)
