import sys
sys.dont_write_bytecode=True
import numpy as np
import os

'''
In this script the geometry and mesh we are creating is defined in the function 'Create', with dimensional arguments and mesh arguments passed to it. The 'test' function provides dimensions for when the script is loaded manually in to Salome and not via a parametric study. The error function is imported during the setup of parametric studies to check for any geometrical errors which may arise.
'''


def Create(**kwargs):
	from salome.geom import geomBuilder
	from salome.smesh import smeshBuilder
	import  SMESH
	import salome_version
	import SalomeFunc

	if salome_version.getVersions()[0] < 9:
		import salome
		theStudy = salome.myStudy
		geompy = geomBuilder.New(theStudy)
		smesh = smeshBuilder.New(theStudy)
	else :
		geompy = geomBuilder.New()
		smesh = smeshBuilder.New()

	Parameter = kwargs['Parameter']
	MeshFile = kwargs['MeshFile']

	###
	### GEOM component
	###
	print('Creating Geometry')
	# Width and Length of the transition from gauge to handle
	TransWidth = (Parameter.HandleWidth - Parameter.GaugeWidth)/2
	TransLength = (TransWidth*(2*Parameter.TransRad - TransWidth))**0.5

	# Get major and minor diameter for ellipse
	Diam_a = 2*Parameter.Rad_a
	Diam_b = 2*Parameter.Rad_b

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
	Vertex_2 = geompy.MakeVertex(0, Parameter.HandleWidth, 0)
	Vertex_3 = geompy.MakeVertex(Parameter.HandleLength, Parameter.HandleWidth, 0)
	Vertex_4 = geompy.MakeVertex(Parameter.HandleLength, 0, 0)
	Vertex_5 = geompy.MakeVertexWithRef(Vertex_4, TransLength, TransWidth, 0)
	Vertex_6 = geompy.MakeVertexWithRef(Vertex_5, 0, Parameter.GaugeWidth, 0)
	Vertex_7 = geompy.MakeVertexWithRef(Vertex_5, Parameter.GaugeLength, Parameter.GaugeWidth, 0)
	Vertex_8 = geompy.MakeVertexWithRef(Vertex_5, Parameter.GaugeLength, 0, 0)
	Vertex_9 = geompy.MakeVertexWithRef(Vertex_7, TransLength, TransWidth, 0)
	Vertex_10 = geompy.MakeVertexWithRef(Vertex_9, 0, -Parameter.HandleWidth, 0)
	Vertex_11 = geompy.MakeVertexWithRef(Vertex_9, Parameter.HandleLength, -Parameter.HandleWidth, 0)
	Vertex_12 = geompy.MakeVertexWithRef(Vertex_9, Parameter.HandleLength, 0, 0)
	Line_1 = geompy.MakeLineTwoPnt(Vertex_1, Vertex_2)
	Line_2 = geompy.MakeLineTwoPnt(Vertex_2, Vertex_3)
	Line_3 = geompy.MakeLineTwoPnt(Vertex_1, Vertex_4)
	Line_4 = geompy.MakeLineTwoPnt(Vertex_5, Vertex_8)
	Line_5 = geompy.MakeLineTwoPnt(Vertex_6, Vertex_7)
	Line_6 = geompy.MakeLineTwoPnt(Vertex_9, Vertex_12)
	Line_7 = geompy.MakeLineTwoPnt(Vertex_10, Vertex_11)
	Line_8 = geompy.MakeLineTwoPnt(Vertex_11, Vertex_12)

	# Vertices of centre of circle to create arc between handle and gauge
	Vertex_13 = geompy.MakeVertexWithRef(Vertex_5, 0, -Parameter.TransRad, 0)
	Vertex_14 = geompy.MakeVertexWithRef(Vertex_6, 0, Parameter.TransRad, 0)
	Vertex_15 = geompy.MakeVertexWithRef(Vertex_7, 0, Parameter.TransRad, 0)
	Vertex_16 = geompy.MakeVertexWithRef(Vertex_8, 0, -Parameter.TransRad, 0)
	Arc_1 = geompy.MakeArcCenter(Vertex_13, Vertex_4, Vertex_5,False)
	Arc_2 = geompy.MakeArcCenter(Vertex_14, Vertex_3, Vertex_6,False)
	Arc_3 = geompy.MakeArcCenter(Vertex_15, Vertex_7, Vertex_9,False)
	Arc_4 = geompy.MakeArcCenter(Vertex_16, Vertex_8, Vertex_10,False)

	# Combine lines and arcs to create a 2D face
	Wire_1 = geompy.MakeWire([Line_1, Line_2, Line_3, Line_4, Line_5, Line_6, Line_7, Line_8, Arc_1, Arc_2, Arc_3, Arc_4], 1e-05)
	Face_1 = geompy.MakeFaceWires([Wire_1], 1)
	# extrude face to 3D
	Full = geompy.MakePrismVecH(Face_1, OZ, Parameter.Thickness)

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
	if Parameter.Rad_a != 0:
		Vertex_17 = geompy.MakeVertex(Parameter.HandleLength + TransLength + Parameter.GaugeLength/2 + Parameter.HoleCentre[0], Parameter.HandleWidth/2 + Parameter.HoleCentre[1], 0)
		if Parameter.Rad_a >= Parameter.Rad_b:
			Ellipse_1 = geompy.MakeEllipse(Vertex_17, OZ, Diam_a, Diam_b, OX)
		if Parameter.Rad_a < Parameter.Rad_b:
			Ellipse_1 = geompy.MakeEllipse(Vertex_17, OZ, Diam_b, Diam_a, OY)
		NotchFace = geompy.MakeFaceWires([Ellipse_1], 1)
		Notch = geompy.MakePrismVecH(NotchFace, OZ, Parameter.Thickness)
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

	if Parameter.Rad_a != 0:
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
	Local_Length_1 = Regular_1D_1.LocalLength(Parameter.Length1D,None,1e-07)
	NETGEN_2D_1 = Mesh_1.Triangle(algo=smeshBuilder.NETGEN_2D)
	NETGEN_2D_Parameters_1 = NETGEN_2D_1.Parameters()
	NETGEN_2D_Parameters_1.SetMaxSize( Parameter.Length2D )
	NETGEN_2D_Parameters_1.SetOptimize( 1 )
	NETGEN_2D_Parameters_1.SetFineness( 3 )
	NETGEN_2D_Parameters_1.SetChordalError( 0.1 )
	NETGEN_2D_Parameters_1.SetChordalErrorEnabled( 0 )
	NETGEN_2D_Parameters_1.SetMinSize( Parameter.Length2D*0.5 )
	NETGEN_2D_Parameters_1.SetUseSurfaceCurvature( 1 )
	NETGEN_2D_Parameters_1.SetQuadAllowed( 0 )
	NETGEN_3D_1 = Mesh_1.Tetrahedron()
	NETGEN_3D_Parameters_1 = NETGEN_3D_1.Parameters()
	NETGEN_3D_Parameters_1.SetMaxSize( Parameter.Length3D )
	NETGEN_3D_Parameters_1.SetOptimize( 1 )
	NETGEN_3D_Parameters_1.SetFineness( 3 )
	NETGEN_3D_Parameters_1.SetMinSize( Parameter.Length3D*0.5 )

	smesh.SetName(Mesh_1.GetMesh(), Parameter.Name)
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
	if Parameter.Rad_a != 0:
		## Calculate circumference of the hole using Ramanujan approximation
		HoleCirc = np.pi*(3*(Diam_a + Diam_b) - ((3*Diam_a + Diam_b)*(Diam_a + 3*Diam_b))**0.5)
		HoleLength = HoleCirc/Parameter.HoleDisc

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
	if MeshFile:
		from SalomeFunc import MeshExport
		MeshExport(Mesh_1,MeshFile)

	globals().update(locals())


def error(Parameters):
	''' This function is imported in during the Setup to pick up any errors which will occur for the given geometrical dimension. i.e. impossible dimensions '''

	message = None
	if Parameters.HandleWidth > (Parameters.GaugeWidth + 2*Parameters.TransRad):
		message = 'Error: Handle width too wide for given gauge width and arc radius ({})'.format(Parameters.MeshName)
	if (Parameters.Rad_a == 0 and Parameters.Rad_b !=0) or (Parameters.Rad_a != 0 and Parameters.Rad_b ==0) : 
		message = 'Error: Both Parameter.Rad_a and Parameter.Rad_b must both be zero or non-zero ({})'.format(Parameters.MeshName)
	if (Parameters.Rad_a < 0 or Parameters.Rad_b < 0): 
		message = 'Error: Radii must be positive ({})'.format(Parameters.MeshName)
	if abs(Parameters.HoleCentre[1]) + 2*Parameters.Rad_b >= Parameters.GaugeWidth/2:
		message = 'Error: Hole not entirely in testpiece ({})'.format(Parameters.MeshName)
	if abs(Parameters.HoleCentre[0]) + 2*Parameters.Rad_a >= Parameters.GaugeLength/2:
		message = 'Error: Hole not entirely in gauge ({})'.format(Parameters.MeshName)
	
	return message

class TestDimensions():
	def __init__(self):
		### Geometric parameters
		self.Thickness = 0.003
		self.HandleWidth = 0.036
		self.HandleLength = 0.024
		self.GaugeWidth = 0.012
		self.GaugeLength = 0.04
		self.TransRad = 0.012
		self.HoleCentre = (0.00,0.00)
		self.Rad_a = 0.001
		self.Rad_b = 0.0005
		### Mesh parameters
		self.HoleDisc = 50
		self.Length1D = 0.002
		self.Length2D = 0.002
		self.Length3D = 0.002
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


