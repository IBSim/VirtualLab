import sys
sys.dont_write_bytecode=True
import numpy as np
import os
from types import SimpleNamespace
from Scripts.Common.VLFunctions import VerifyParameters

'''
This script generates a 'dog bone' sample using the SALOME software package.
This sample is commonly used to perform tensile tests, which can also be carried out
as part of the VirtualLab package. An optional hole can be included in the sample.
An image of the dog bone sample and reference to variable names can be found in:
https://gitlab.com/ibsim/media/-/blob/master/images/VirtualLab/DogBone.png

The function 'Example' contains the variables and values required to generate a
mesh.
'''

def Example():
	'''
	Example parameter values.
	'''
	Parameters = SimpleNamespace(Name='Test')
	# === Geometrical dimensions ===
	Parameters.Thickness = 0.003
	Parameters.HandleWidth = 0.036
	Parameters.HandleLength = 0.024
	Parameters.GaugeWidth = 0.012
	Parameters.GaugeLength = 0.04
	Parameters.TransRad = 0.012
	Parameters.HoleCentre = (0.00,0.00)
	Parameters.Rad_a = 0.001
	Parameters.Rad_b = 0.0005

	# === Mesh sizes ===
	Parameters.HoleDisc = 50
	Parameters.Length1D = 0.002
	Parameters.Length2D = 0.002
	Parameters.Length3D = 0.002

	return Parameters

def Verify(Parameters):
	''''
	Verify that the parameters set in Parameters_Master and/or Parameters_Var
	are suitable to create the mesh.
	These can either be a warning or an error
	'''
	error, warning = [],[]

	# =============================================================
	# Check Variables are defined in the parameters

	# Required variables
	ReqVar = ['Thickness','HandleWidth',
			  'HandleLength','GaugeWidth',
			  'GaugeLength','TransRad',
			  'Length1D','Length2D','Length3D']
	# Optional variables - all are needed to create a hole
	OptVar = ['HoleCentre','Rad_a','Rad_b','HoleDisc']

	miss = VerifyParameters(Parameters,ReqVar)
	if miss:
		error.append("The following variables are not declared in the "\
					 "mesh parameters:\n{}".format("\n".join(miss)))

	miss = VerifyParameters(Parameters,OptVar)
	if miss and len(miss)<len(OptVar):
		error.append("The following variables are not declared in the "\
					 "mesh parameters:\n{}".format("\n".join(miss)))
	if error:
		return error, warning

	# =============================================================
	# Check conditons based on the dimensions provided
	if Parameters.HandleWidth > (Parameters.GaugeWidth + 2*Parameters.TransRad):
		error.append('Handle width too wide for given gauge width and arc radius')

	if hasattr(Parameters,'Rad_a'):
		if (Parameters.Rad_a == 0 and Parameters.Rad_b !=0) or (Parameters.Rad_a != 0 and Parameters.Rad_b ==0):
			error.append('Both Parameters.Rad_a and Parameters.Rad_b must both be zero or non-zero')
		if (Parameters.Rad_a < 0 or Parameters.Rad_b < 0):
			error.append('Radii must be positive')
		if abs(Parameters.HoleCentre[1]) + 2*Parameters.Rad_b >= Parameters.GaugeWidth/2:
			error.append('Hole not entirely in testpiece')
		if abs(Parameters.HoleCentre[0]) + 2*Parameters.Rad_a >= Parameters.GaugeLength/2:
			warning.append('Hole not entirely in gauge')

	return error,warning

def Create(Parameters):
	from salome.geom import geomBuilder
	from salome.smesh import smeshBuilder
	import  SMESH
	import salome_version
	from Scripts.Common.VLPackages.Salome import SalomeFunc

	if salome_version.getVersions()[0] < 9:
		import salome
		theStudy = salome.myStudy
		geompy = geomBuilder.New(theStudy)
		smesh = smeshBuilder.New(theStudy)
	else :
		geompy = geomBuilder.New()
		smesh = smeshBuilder.New()

	Hole = hasattr(Parameters,'Rad_a')

	###
	### GEOM component
	###
	print('Creating Geometry')
	# Width and Length of the transition from gauge to handle
	TransWidth = (Parameters.HandleWidth - Parameters.GaugeWidth)/2
	TransLength = (TransWidth*(2*Parameters.TransRad - TransWidth))**0.5

	O = geompy.MakeVertex(0, 0, 0)
	OX = geompy.MakeVectorDXDYDZ(1, 0, 0)
	OY = geompy.MakeVectorDXDYDZ(0, 1, 0)
	OZ = geompy.MakeVectorDXDYDZ(0, 0, 1)
	geompy.addToStudy( O, 'O' )
	geompy.addToStudy( OX, 'OX' )
	geompy.addToStudy( OY, 'OY' )
	geompy.addToStudy( OZ, 'OZ' )

	# Vertexes of shape and connecting lines
	Vertex_1 = geompy.MakeVertex(0, 0, 0)
	Vertex_2 = geompy.MakeVertex(0, Parameters.HandleWidth, 0)
	Vertex_3 = geompy.MakeVertex(Parameters.HandleLength, Parameters.HandleWidth, 0)
	Vertex_4 = geompy.MakeVertex(Parameters.HandleLength, 0, 0)
	Vertex_5 = geompy.MakeVertexWithRef(Vertex_4, TransLength, TransWidth, 0)
	Vertex_6 = geompy.MakeVertexWithRef(Vertex_5, 0, Parameters.GaugeWidth, 0)
	Vertex_7 = geompy.MakeVertexWithRef(Vertex_5, Parameters.GaugeLength, Parameters.GaugeWidth, 0)
	Vertex_8 = geompy.MakeVertexWithRef(Vertex_5, Parameters.GaugeLength, 0, 0)
	Vertex_9 = geompy.MakeVertexWithRef(Vertex_7, TransLength, TransWidth, 0)
	Vertex_10 = geompy.MakeVertexWithRef(Vertex_9, 0, -Parameters.HandleWidth, 0)
	Vertex_11 = geompy.MakeVertexWithRef(Vertex_9, Parameters.HandleLength, -Parameters.HandleWidth, 0)
	Vertex_12 = geompy.MakeVertexWithRef(Vertex_9, Parameters.HandleLength, 0, 0)
	Line_1 = geompy.MakeLineTwoPnt(Vertex_1, Vertex_2)
	Line_2 = geompy.MakeLineTwoPnt(Vertex_2, Vertex_3)
	Line_3 = geompy.MakeLineTwoPnt(Vertex_1, Vertex_4)
	Line_4 = geompy.MakeLineTwoPnt(Vertex_5, Vertex_8)
	Line_5 = geompy.MakeLineTwoPnt(Vertex_6, Vertex_7)
	Line_6 = geompy.MakeLineTwoPnt(Vertex_9, Vertex_12)
	Line_7 = geompy.MakeLineTwoPnt(Vertex_10, Vertex_11)
	Line_8 = geompy.MakeLineTwoPnt(Vertex_11, Vertex_12)

	# Vertices of centre of circle to create arc between handle and gauge
	Vertex_13 = geompy.MakeVertexWithRef(Vertex_5, 0, -Parameters.TransRad, 0)
	Vertex_14 = geompy.MakeVertexWithRef(Vertex_6, 0, Parameters.TransRad, 0)
	Vertex_15 = geompy.MakeVertexWithRef(Vertex_7, 0, Parameters.TransRad, 0)
	Vertex_16 = geompy.MakeVertexWithRef(Vertex_8, 0, -Parameters.TransRad, 0)
	Arc_1 = geompy.MakeArcCenter(Vertex_13, Vertex_4, Vertex_5,False)
	Arc_2 = geompy.MakeArcCenter(Vertex_14, Vertex_3, Vertex_6,False)
	Arc_3 = geompy.MakeArcCenter(Vertex_15, Vertex_7, Vertex_9,False)
	Arc_4 = geompy.MakeArcCenter(Vertex_16, Vertex_8, Vertex_10,False)

	# Combine lines and arcs to create a 2D face
	Wire_1 = geompy.MakeWire([Line_1, Line_2, Line_3, Line_4, Line_5, Line_6, Line_7, Line_8, Arc_1, Arc_2, Arc_3, Arc_4], 1e-05)
	Face_1 = geompy.MakeFaceWires([Wire_1], 1)
	# extrude face to 3D
	Full = geompy.MakePrismVecH(Face_1, OZ, Parameters.Thickness)

	geompy.addToStudy( Vertex_1, 'Vertex_1' )
	geompy.addToStudy( Vertex_2, 'Vertex_2' )
	geompy.addToStudy( Vertex_3, 'Vertex_3' )
	geompy.addToStudy( Vertex_4, 'Vertex_4' )
	geompy.addToStudy( Vertex_5, 'Vertex_5' )
	geompy.addToStudy( Vertex_6, 'Vertex_6' )
	geompy.addToStudy( Vertex_7, 'Vertex_7' )
	geompy.addToStudy( Vertex_8, 'Vertex_8' )
	geompy.addToStudy( Vertex_9, 'Vertex_9' )
	geompy.addToStudy( Vertex_10, 'Vertex_10' )
	geompy.addToStudy( Vertex_11, 'Vertex_11' )
	geompy.addToStudy( Vertex_12, 'Vertex_12' )
	geompy.addToStudy( Vertex_13, 'Vertex_13' )
	geompy.addToStudy( Vertex_14, 'Vertex_14' )
	geompy.addToStudy( Vertex_15, 'Vertex_15' )
	geompy.addToStudy( Vertex_16, 'Vertex_16' )
	geompy.addToStudy( Line_1, 'Line_1' )
	geompy.addToStudy( Line_2, 'Line_2' )
	geompy.addToStudy( Line_3, 'Line_3' )
	geompy.addToStudy( Line_4, 'Line_4' )
	geompy.addToStudy( Line_5, 'Line_5' )
	geompy.addToStudy( Line_6, 'Line_6' )
	geompy.addToStudy( Line_7, 'Line_7' )
	geompy.addToStudy( Line_8, 'Line_8' )
	geompy.addToStudy( Arc_1, 'Arc_1' )
	geompy.addToStudy( Arc_2, 'Arc_2' )
	geompy.addToStudy( Arc_3, 'Arc_3' )
	geompy.addToStudy( Arc_4, 'Arc_4' )
	geompy.addToStudy( Wire_1, 'Wire_1' )
	geompy.addToStudy( Face_1, 'Face_1' )
	geompy.addToStudy( Full, 'Full' )

	# If Radius is non-zero we will create the shape of the notch and then cut it from the sample
	if Hole:
		# Get major and minor diameter for ellipse
		Diam_a = 2*Parameters.Rad_a
		Diam_b = 2*Parameters.Rad_b

		Vertex_17 = geompy.MakeVertex(Parameters.HandleLength + TransLength + Parameters.GaugeLength/2 + Parameters.HoleCentre[0], Parameters.HandleWidth/2 + Parameters.HoleCentre[1], 0)
		if Parameters.Rad_a >= Parameters.Rad_b:
			Ellipse_1 = geompy.MakeEllipse(Vertex_17, OZ, Diam_a, Diam_b, OX)
		if Parameters.Rad_a < Parameters.Rad_b:
			Ellipse_1 = geompy.MakeEllipse(Vertex_17, OZ, Diam_b, Diam_a, OY)
		NotchFace = geompy.MakeFaceWires([Ellipse_1], 1)
		Notch = geompy.MakePrismVecH(NotchFace, OZ, Parameters.Thickness)
		Testpiece = geompy.MakeCutList(Full, [Notch], True)

		geompy.addToStudy( NotchFace, 'NotchFace' )
		geompy.addToStudy( Notch, 'Notch' )
	else :
		Testpiece = Full

	geompy.addToStudy( Testpiece, 'Testpiece' )

	### Create Groups
	# SalomeFunc.ObjIndex returns the SubShape index for one shape given the shape and index of the parent shape
	Ix = SalomeFunc.ObjIndex(Testpiece, Full, [3])[0]
	Constrain = geompy.CreateGroup(Testpiece, geompy.ShapeType["FACE"])
	geompy.UnionIDs(Constrain, Ix)
	geompy.addToStudyInFather( Testpiece, Constrain, 'Constrain' )

	Ix = SalomeFunc.ObjIndex(Testpiece, Full, [48])[0]
	Load = geompy.CreateGroup(Testpiece, geompy.ShapeType["FACE"])
	geompy.UnionIDs(Load, Ix)
	geompy.addToStudyInFather( Testpiece, Load, 'Load' )

	Ix = SalomeFunc.ObjIndex(Testpiece, Full, [6])[0]
	Constrain3 = geompy.CreateGroup(Testpiece, geompy.ShapeType["VERTEX"])
	geompy.UnionIDs(Constrain3, Ix)
	geompy.addToStudyInFather(Testpiece, Constrain3, 'Constrain3' )

	Ix = SalomeFunc.ObjIndex(Testpiece, Full, [7])[0]
	Constrain2 = geompy.CreateGroup(Testpiece, geompy.ShapeType["VERTEX"])
	geompy.UnionIDs(Constrain2, Ix)
	geompy.addToStudyInFather( Testpiece, Constrain2, 'Constrain2' )

	Ix = SalomeFunc.ObjIndex(Testpiece, Full, [10])[0]
	Constrain1 = geompy.CreateGroup(Testpiece, geompy.ShapeType["VERTEX"])
	geompy.UnionIDs(Constrain1, Ix)
	geompy.addToStudyInFather( Testpiece, Constrain1, 'Constrain1' )

	if Hole:
		Ix = SalomeFunc.ObjIndex(Testpiece, Notch, [3])[0]
		Notch_Surf = geompy.CreateGroup(Testpiece, geompy.ShapeType["FACE"])
		geompy.UnionIDs(Notch_Surf, Ix)
		geompy.addToStudyInFather( Testpiece, Notch_Surf, 'Notch_Surf' )

		Ix = SalomeFunc.ObjIndex(Testpiece, Notch, [8, 9])[0]
		Notch_Edge = geompy.CreateGroup(Testpiece, geompy.ShapeType["EDGE"])
		geompy.UnionIDs(Notch_Edge, Ix)
		geompy.addToStudyInFather( Testpiece, Notch_Edge, 'Notch_Edge' )


	###
	### SMESH component
	###
	print('Creating Mesh')
	### Main mesh
	Mesh_1 = smesh.Mesh(Testpiece)
	Regular_1D_1 = Mesh_1.Segment()
	Local_Length_1 = Regular_1D_1.LocalLength(Parameters.Length1D,None,1e-07)
	NETGEN_2D_1 = Mesh_1.Triangle(algo=smeshBuilder.NETGEN_2D)
	NETGEN_2D_Parameters_1 = NETGEN_2D_1.Parameters()
	NETGEN_2D_Parameters_1.SetMaxSize( Parameters.Length2D )
	NETGEN_2D_Parameters_1.SetOptimize( 1 )
	NETGEN_2D_Parameters_1.SetFineness( 3 )
	NETGEN_2D_Parameters_1.SetChordalError( 0.1 )
	NETGEN_2D_Parameters_1.SetChordalErrorEnabled( 0 )
	NETGEN_2D_Parameters_1.SetMinSize( Parameters.Length2D*0.5 )
	NETGEN_2D_Parameters_1.SetUseSurfaceCurvature( 1 )
	NETGEN_2D_Parameters_1.SetQuadAllowed( 0 )
	NETGEN_3D_1 = Mesh_1.Tetrahedron()
	NETGEN_3D_Parameters_1 = NETGEN_3D_1.Parameters()
	NETGEN_3D_Parameters_1.SetMaxSize( Parameters.Length3D )
	NETGEN_3D_Parameters_1.SetOptimize( 1 )
	NETGEN_3D_Parameters_1.SetFineness( 3 )
	NETGEN_3D_Parameters_1.SetMinSize( Parameters.Length3D*0.5 )

	smesh.SetName(Mesh_1.GetMesh(), Parameters.Name)
	smesh.SetName(Regular_1D_1.GetAlgorithm(), 'Regular_1D_1')
	smesh.SetName(NETGEN_2D_1.GetAlgorithm(), 'NETGEN 2D_1')
	smesh.SetName(NETGEN_3D_1.GetAlgorithm(), 'NETGEN 3D_1')
	smesh.SetName(Local_Length_1, 'Local Length_1')
	smesh.SetName(NETGEN_2D_Parameters_1, 'NETGEN 2D Parameters_1')
	smesh.SetName(NETGEN_3D_Parameters_1, 'NETGEN 3D Parameters_1')

	### Add groups
	F_Load = Mesh_1.GroupOnGeom(Load,'Load',SMESH.FACE)
	N_Load = Mesh_1.GroupOnGeom(Load,'NLoad',SMESH.NODE)

	F_Constrain = Mesh_1.GroupOnGeom(Constrain,'Constrain',SMESH.FACE)
	N_Constrain = Mesh_1.GroupOnGeom(Constrain,'NConstrain',SMESH.NODE)

	N_Constrain3 = Mesh_1.GroupOnGeom(Constrain3,'Constrain3',SMESH.NODE)
	N_Constrain2 = Mesh_1.GroupOnGeom(Constrain2,'Constrain2',SMESH.NODE)
	N_Constrain1 = Mesh_1.GroupOnGeom(Constrain1,'Constrain1',SMESH.NODE)

	### SubMesh 1 - Refinement near the hole
	if Hole:
		## Calculate circumference of the hole using Ramanujan approximation
		HoleCirc = np.pi*(3*(Diam_a + Diam_b) - ((3*Diam_a + Diam_b)*(Diam_a + 3*Diam_b))**0.5)
		HoleLength = HoleCirc/Parameters.HoleDisc

		Regular_1D_2 = Mesh_1.Segment(geom=Notch_Surf)
		Local_Length_2 = Regular_1D_2.LocalLength(HoleLength,None,1e-07)
		NETGEN_2D_2 = Mesh_1.Triangle(algo=smeshBuilder.NETGEN_2D,geom=Notch_Surf)
		NETGEN_2D_Parameters_2 = NETGEN_2D_2.Parameters()
		NETGEN_2D_Parameters_2.SetMaxSize( HoleLength )
		NETGEN_2D_Parameters_2.SetOptimize( 1 )
		NETGEN_2D_Parameters_2.SetFineness( 3 )
		NETGEN_2D_Parameters_2.SetChordalError( 0.1 )
		NETGEN_2D_Parameters_2.SetChordalErrorEnabled( 0 )
		NETGEN_2D_Parameters_2.SetMinSize( HoleLength )
		NETGEN_2D_Parameters_2.SetUseSurfaceCurvature( 1 )
		NETGEN_2D_Parameters_2.SetQuadAllowed( 0 )
		Sub_mesh_1 = Regular_1D_2.GetSubMesh()

		smesh.SetName(Sub_mesh_1, 'Sub-mesh_1')
		smesh.SetName(Regular_1D_2.GetAlgorithm(), 'Regular_1D_2')
		smesh.SetName(NETGEN_2D_2.GetAlgorithm(), 'NETGEN 2D_2')
		smesh.SetName(Local_Length_2, 'Local Length_2')
		smesh.SetName(NETGEN_2D_Parameters_2, 'NETGEN 2D Parameters_2')

		NETGEN_2D_Parameters_1.SetMinSize( HoleLength )
		NETGEN_3D_Parameters_1.SetMinSize( HoleLength )

		# Add Groups
		F_NotchSurf = Mesh_1.GroupOnGeom(Notch_Surf,'Notch_Surf',SMESH.FACE)

	isDone = Mesh_1.Compute()

	globals().update(locals())

	return Mesh_1

if __name__ == '__main__':
	if len(sys.argv) == 1:
		Create(Example())
	# 1 argument provided which is the parameter file
	elif len(sys.argv) == 2:
		ParameterFile = sys.argv[1]
		sys.path.insert(0, os.path.dirname(ParameterFile))
		Parameters = __import__(os.path.splitext(os.path.basename(ParameterFile))[0])
		Create(Parameters)
