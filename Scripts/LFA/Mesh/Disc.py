import sys
sys.dont_write_bytecode=True
import numpy as np
import os

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

	if Parameter.VoidRadius and Parameter.VoidHeight:
		isVoid = True
	else:
		isVoid = False

	###
	### GEOM component
	###
	O = geompy.MakeVertex(0, 0, 0)
	OX = geompy.MakeVectorDXDYDZ(1, 0, 0)
	OY = geompy.MakeVectorDXDYDZ(0, 1, 0)
	OZ = geompy.MakeVectorDXDYDZ(0, 0, 1)
	geompy.addToStudy( O, 'O' )
	geompy.addToStudy( OX, 'OX' )
	geompy.addToStudy( OY, 'OY' )
	geompy.addToStudy( OZ, 'OZ' )

	Cylinder_b = geompy.MakeCylinder(O, OZ, Parameter.Radius, Parameter.HeightB)
	Vertex_1 = geompy.MakeVertexWithRef(O, 0, 0, Parameter.HeightB)
	Cylinder_t = geompy.MakeCylinder(Vertex_1, OZ, Parameter.Radius, Parameter.HeightT)
	Cylinder_ext = geompy.MakeCylinder(O, OZ, Parameter.Radius, Parameter.HeightB + Parameter.HeightT)
	Cylinder_ext = geompy.MakePartition([Cylinder_ext], [Cylinder_b], [], [], geompy.ShapeType["SOLID"], 0, [], 0)
	geompy.addToStudy(Cylinder_ext,'Cylinder_ext')
	geompy.addToStudy(Cylinder_b,'Cylinder_b')
	geompy.addToStudy(Cylinder_t,'Cylinder_t')


	if isVoid:
		Vertex_2 = geompy.MakeVertexWithRef(O, Parameter.VoidCentre[0], Parameter.VoidCentre[1], Parameter.HeightB)
		if Parameter.VoidHeight > 0:
			vect = OZ
		else :
			vect = geompy.MakeVectorDXDYDZ(0, 0, -1)
		Void = geompy.MakeCylinder(Vertex_2, vect, Parameter.VoidRadius, abs(Parameter.VoidHeight))
		geompy.addToStudy(Void, 'Void')
		Testpiece = geompy.MakeCutList(Cylinder_ext,[Void], True)
	else :
		Testpiece = Cylinder_ext

	geompy.addToStudy(Testpiece, 'Testpiece')

	### Add solid groups
	Vollst = geompy.SubShapeAllIDs(Testpiece,geompy.ShapeType['SOLID'])
	DiskT = geompy.CreateGroup(Testpiece, geompy.ShapeType["SOLID"])
	geompy.UnionIDs(DiskT, [Vollst[0]])
	geompy.addToStudyInFather( Testpiece, DiskT, 'Top' )

	DiskB = geompy.CreateGroup(Testpiece, geompy.ShapeType["SOLID"])
	geompy.UnionIDs(DiskB, [Vollst[1]])
	geompy.addToStudyInFather( Testpiece, DiskB, 'Bottom' )


	### Add Faces
	# Top_Face - The face which the laser pulse will be applied
	# Find Ix of the face corresponding to number 10 in Cylinder_t
	Ix = SalomeFunc.ObjIndex(Testpiece, Cylinder_t, [10])[0]
	Top_Face = geompy.CreateGroup(Testpiece, geompy.ShapeType["FACE"])
	geompy.UnionIDs(Top_Face, Ix)
	geompy.addToStudyInFather( Testpiece, Top_Face, 'Top_Face' )

	# Top_Ext - External surfaces for the top disk for HTC
	# Find Ix of the face corresponding to number 10 and 3 in Cylinder_t
	Ix = SalomeFunc.ObjIndex(Testpiece, Cylinder_t, [10, 3])[0]
	Top_Ext = geompy.CreateGroup(Testpiece, geompy.ShapeType["FACE"])
	geompy.UnionIDs(Top_Ext, Ix)
	geompy.addToStudyInFather( Testpiece, Top_Ext, 'Top_Ext' )

	# Bottom_Face - Used to measure thermal conductivity during post processing
	# Find the face corresponding to number 12 in Cylinder_b
	Ix = SalomeFunc.ObjIndex(Testpiece, Cylinder_b, [12])[0]
	Bottom_Face = geompy.CreateGroup(Testpiece, geompy.ShapeType["FACE"])
	geompy.UnionIDs(Bottom_Face, Ix)
	geompy.addToStudyInFather( Testpiece, Bottom_Face, 'Bottom_Face' )

	# Bottom_Ext - External surfaces for the bottom disk for HTC
	# Find the face corresponding to number 3 and 12 in Cylinder_b
	Ix = SalomeFunc.ObjIndex(Testpiece, Cylinder_b, [3, 12])[0]
	Bottom_Ext = geompy.CreateGroup(Testpiece, geompy.ShapeType["FACE"])
	geompy.UnionIDs(Bottom_Ext, Ix)
	geompy.addToStudyInFather( Testpiece, Bottom_Ext, 'Bottom_Ext' )

	if isVoid:
		# VoidExt - Find the faces corresponding to 3(side), 10(top) and 12(bottom) in Void
		Ix = SalomeFunc.ObjIndex(Testpiece, Void, [3, 10, 12])[0]
		VoidExt = geompy.CreateGroup(Testpiece, geompy.ShapeType["FACE"])
		geompy.UnionIDs(VoidExt, Ix)
		geompy.addToStudyInFather( Testpiece, VoidExt, 'Void_Ext' )

	###
	### SMESH component
	###

	### Create Main Mesh
	Mesh_1 = smesh.Mesh(Testpiece)
	Regular_1D = Mesh_1.Segment()
	Local_Length_1 = Regular_1D.LocalLength(Parameter.Length1D,None,1e-07)
	NETGEN_2D = Mesh_1.Triangle(algo=smeshBuilder.NETGEN_2D)
	NETGEN_2D_Parameters_1 = NETGEN_2D.Parameters()
	NETGEN_2D_Parameters_1.SetMaxSize( Parameter.Length2D )
	NETGEN_2D_Parameters_1.SetOptimize( 1 )
	NETGEN_2D_Parameters_1.SetFineness( 3 )
	NETGEN_2D_Parameters_1.SetChordalError( 0.1 )
	NETGEN_2D_Parameters_1.SetChordalErrorEnabled( 0 )
	NETGEN_2D_Parameters_1.SetMinSize( Parameter.Length2D )
	NETGEN_2D_Parameters_1.SetUseSurfaceCurvature( 1 )
	NETGEN_2D_Parameters_1.SetQuadAllowed( 0 )
	NETGEN_3D = Mesh_1.Tetrahedron()
	NETGEN_3D_Parameters_1 = NETGEN_3D.Parameters()
	NETGEN_3D_Parameters_1.SetMaxSize( Parameter.Length3D )
	NETGEN_3D_Parameters_1.SetOptimize( 1 )
	NETGEN_3D_Parameters_1.SetFineness( 2 )
	NETGEN_3D_Parameters_1.SetMinSize( Parameter.Length3D )

	smesh.SetName(Regular_1D.GetAlgorithm(), 'Regular_1D')
	smesh.SetName(NETGEN_3D.GetAlgorithm(), 'NETGEN 3D')
	smesh.SetName(NETGEN_2D.GetAlgorithm(), 'NETGEN 2D')
	smesh.SetName(NETGEN_2D_Parameters_1, 'NETGEN 2D Parameters_1')
	smesh.SetName(Local_Length_1, 'Local Length_1')
	smesh.SetName(NETGEN_3D_Parameters_1, 'NETGEN 3D Parameters_1')
	smesh.SetName(Mesh_1, Parameter.Name)

	## Add groups
	Mesh_Top_Ext = Mesh_1.GroupOnGeom(DiskT,'Top',SMESH.VOLUME)
	Mesh_Top_Ext = Mesh_1.GroupOnGeom(DiskB,'Bottom',SMESH.VOLUME)

	Mesh_Top_Ext = Mesh_1.GroupOnGeom(Top_Ext,'Top_Ext',SMESH.FACE)
	Mesh_Bottom_Ext = Mesh_1.GroupOnGeom(Bottom_Ext,'Bottom_Ext',SMESH.FACE)
	Mesh_Top_Face = Mesh_1.GroupOnGeom(Top_Face,'Top_Face',SMESH.FACE)
	Mesh_Bottom_Face = Mesh_1.GroupOnGeom(Bottom_Face,'Bottom_Face',SMESH.FACE)

	Mesh_Bottom_Ext_Node = Mesh_1.GroupOnGeom(Bottom_Face,'NBottom_Face',SMESH.NODE)
	Mesh_Top_Ext_Node = Mesh_1.GroupOnGeom(Top_Face,'NTop_Face',SMESH.NODE)

	if isVoid:
		HoleCirc = 2*np.pi*Parameter.VoidRadius
		HoleLength = HoleCirc/Parameter.VoidDisc

		### Sub Mesh creation
		## Sub-Mesh 2
		Regular_1D_2 = Mesh_1.Segment(geom=VoidExt)
		Local_Length_2 = Regular_1D_2.LocalLength(HoleLength,None,1e-07)
		NETGEN_2D_2 = Mesh_1.Triangle(algo=smeshBuilder.NETGEN_2D,geom=VoidExt)
		NETGEN_2D_Parameters_3 = NETGEN_2D_2.Parameters()
		NETGEN_2D_Parameters_3.SetMaxSize( HoleLength )
		NETGEN_2D_Parameters_3.SetOptimize( 1 )
		NETGEN_2D_Parameters_3.SetFineness( 3 )
		NETGEN_2D_Parameters_3.SetChordalError( 0.1 )
		NETGEN_2D_Parameters_3.SetChordalErrorEnabled( 0 )
		NETGEN_2D_Parameters_3.SetMinSize( HoleLength )
		NETGEN_2D_Parameters_3.SetUseSurfaceCurvature( 1 )
		NETGEN_2D_Parameters_3.SetQuadAllowed( 0 )
		Sub_mesh_2 = NETGEN_2D_2.GetSubMesh()

		smesh.SetName(Sub_mesh_2, 'VoidMesh')
		smesh.SetName(NETGEN_2D_Parameters_3, 'NETGEN 2D Parameters_3')

		NETGEN_2D_Parameters_1.SetMinSize( HoleLength )
		NETGEN_3D_Parameters_1.SetMinSize( HoleLength )

		Mesh_Void_Ext = Mesh_1.GroupOnGeom(VoidExt,'Void_Ext',SMESH.FACE)

	isDone = Mesh_1.Compute()

	if MeshFile:
		SalomeFunc.MeshExport(Mesh_1,MeshFile)

	globals().update(locals())

class TestDimensions():
	def __init__(self):
		self.Name = 'TestMesh'
		self.Radius = 0.0063
		self.HeightB = 0.00125
		self.HeightT = 0.00125
		self.VoidCentre = (0,0)
		self.VoidRadius = 0.0005
		self.VoidHeight = -0.0001
		### Mesh parameters
		self.Length1D = 0.0004 #Length on 1D edges
		self.Length2D = 0.0004 #Maximum length of any edge belonging to a face
		self.Length3D = 0.0004 #Maximum length of any edge belogining to a tetrahedron
		self.VoidDisc = 20
		self.MeshName = 'Test'


def GeomError(Parameters):
	''' This function is imported in during the Setup to pick up any errors which will occur for the given geometrical dimension. i.e. impossible dimensions '''

	message = None
	if Parameters.VoidHeight >= Parameters.HeightT:
		message = 'Void height too large'
	if Parameters.VoidRadius >= Parameters.Radius:
		message = 'Void radius too large'
	if (Parameters.VoidHeight == 0 and  Parameters.VoidRadius !=0) or (Parameters.VoidHeight != 0 and  Parameters.VoidRadius ==0):
		message = 'The height and radius of the void must both be zero or non-zero'
	return message

if __name__ == '__main__':
	if len(sys.argv) == 1:
		Create(Parameter = TestDimensions(),MeshFile = None)
	# 1 argument provided which is the parameter file
	elif len(sys.argv) == 2:
		ParameterFile = sys.argv[1]
		sys.path.insert(0, os.path.dirname(ParameterFile))
		Parameters = __import__(os.path.splitext(os.path.basename(ParameterFile))[0])
		Create(Parameter = Parameters,MeshFile = None)
