import sys
sys.dont_write_bytecode=True
import numpy as np
import os

def Create(Parameter):
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


	## Bottom half of dics
	Cylinder_b_orig = geompy.MakeCylinder(O, OZ, Parameter.Radius, Parameter.HeightB)
	geompy.addToStudy(Cylinder_b_orig,'Cylinder_b_orig')

	# add groups
	Bottom_Face = geompy.CreateGroup(Cylinder_b_orig, geompy.ShapeType["FACE"])
	geompy.UnionIDs(Bottom_Face, [12])
	geompy.addToStudyInFather( Cylinder_b_orig, Bottom_Face, 'Bottom_Face' )

	Bottom_Ext = geompy.CreateGroup(Cylinder_b_orig, geompy.ShapeType["FACE"])
	geompy.UnionIDs(Bottom_Ext, [12,3])
	geompy.addToStudyInFather( Cylinder_b_orig, Bottom_Ext, 'Bottom_Ext' )


	## Top half of disc
	Vertex_1 = geompy.MakeVertexWithRef(O, 0, 0, Parameter.HeightB)
	Cylinder_t_orig = geompy.MakeCylinder(Vertex_1, OZ, Parameter.Radius, Parameter.HeightT)
	geompy.addToStudy(Cylinder_t_orig,'Cylinder_t_orig')

	# add groups
	Top_Face = geompy.CreateGroup(Cylinder_t_orig, geompy.ShapeType["FACE"])
	geompy.UnionIDs(Top_Face, [10])
	geompy.addToStudyInFather( Cylinder_t_orig, Top_Face, 'Top_Face' )

	Top_Ext = geompy.CreateGroup(Cylinder_t_orig, geompy.ShapeType["FACE"])
	geompy.UnionIDs(Top_Ext, [3,10])
	geompy.addToStudyInFather( Cylinder_t_orig, Top_Ext, 'Top_Ext' )

	#Name these with extension _orig since we may change them depending on the
	#location of the void


	# Add void in to disc if needed
	if Parameter.VoidHeight > 0:
		# Void is in the top half of the disc
		Vertex_2 = geompy.MakeVertexWithRef(O, Parameter.VoidCentre[0], Parameter.VoidCentre[1], Parameter.HeightB)
		Void = geompy.MakeCylinder(Vertex_2, OZ, Parameter.VoidRadius, Parameter.VoidHeight)
		Cylinder_t = geompy.MakeCutList(Cylinder_t_orig,[Void], True)
		Cylinder_b = Cylinder_b_orig

	elif Parameter.VoidHeight < 0:
		# Void is in the bottom half of the dics
		Vertex_2 = geompy.MakeVertexWithRef(O, Parameter.VoidCentre[0], Parameter.VoidCentre[1], Parameter.HeightB)
		OZm = geompy.MakeVectorDXDYDZ(0, 0, -1)
		Void = geompy.MakeCylinder(Vertex_2, OZm, Parameter.VoidRadius, -Parameter.VoidHeight)
		Cylinder_b = geompy.MakeCutList(Cylinder_b_orig,[Void], True)
		Cylinder_t = Cylinder_t_orig

	else:
		Cylinder_t = Cylinder_t_orig
		Cylinder_b = Cylinder_b_orig

	# Fuse together top and bottom disk & partition
	Cylinder_full = geompy.MakeFuseList([Cylinder_b, Cylinder_t], True, True)
	Testpiece = geompy.MakePartition([Cylinder_full], [Cylinder_b], [], [], geompy.ShapeType["SOLID"], 0, [], 0)

	geompy.addToStudy(Cylinder_full,'Cylinder_full')
	geompy.addToStudy(Cylinder_b,'Cylinder_b')
	geompy.addToStudy(Cylinder_t,'Cylinder_t')
	geompy.addToStudy(Testpiece, 'Testpiece')


	### Add solid groups
	# List of solid objects IDs. This will have 2 numbers which represent
	# the IDs for the top and bottom disc
	Vollst = geompy.SubShapeAllIDs(Testpiece,geompy.ShapeType['SOLID'])

	# Get geom object & ID of the bottom disc which was used to make the partition
	# This is the same as ObjIndex in salome func
	objDiskB = geompy.GetInPlace(Testpiece, Cylinder_b, False)
	DiskB_ID = objDiskB.GetSubShapeIndices()

	DiskB = geompy.CreateGroup(Testpiece, geompy.ShapeType["SOLID"])
	geompy.UnionIDs(DiskB, DiskB_ID)
	geompy.addToStudyInFather( Testpiece, DiskB, 'Bottom' )

	# The geom object of the top disc is then the remaining ID in Vollst
	Vollst.remove(DiskB_ID[0])
	DiskT = geompy.CreateGroup(Testpiece, geompy.ShapeType["SOLID"])
	geompy.UnionIDs(DiskT, Vollst)
	geompy.addToStudyInFather( Testpiece, DiskT, 'Top' )


	### Add Faces
	# get the necessary groups from Cylinder_b_orig and Cylinder_t_orig
	# and get them as sub shapes of Testpiece

	# Top_Face - The face which the laser pulse will be applied
	Top_Face_ID = Top_Face.GetSubShapeIndices()
	ID = SalomeFunc.ObjIndex(Testpiece, Cylinder_t_orig, Top_Face_ID)[0]
	Top_Face = geompy.CreateGroup(Testpiece, geompy.ShapeType["FACE"])
	geompy.UnionIDs(Top_Face, ID)
	geompy.addToStudyInFather( Testpiece, Top_Face, 'Top_Face' )

	# Top_Ext - External surfaces for the top disk for HTC
	Top_Ext_ID = Top_Ext.GetSubShapeIndices()
	ID = SalomeFunc.ObjIndex(Testpiece, Cylinder_t_orig, Top_Ext_ID)[0]
	Top_Ext = geompy.CreateGroup(Testpiece, geompy.ShapeType["FACE"])
	geompy.UnionIDs(Top_Ext, ID)
	geompy.addToStudyInFather( Testpiece, Top_Ext, 'Top_Ext' )

	# Bottom_Face - Used to measure thermal conductivity during post processing
	Bottom_Face_ID = Bottom_Face.GetSubShapeIndices()
	ID = SalomeFunc.ObjIndex(Testpiece, Cylinder_b_orig, Bottom_Face_ID)[0]
	Bottom_Face = geompy.CreateGroup(Testpiece, geompy.ShapeType["FACE"])
	geompy.UnionIDs(Bottom_Face, ID)
	geompy.addToStudyInFather( Testpiece, Bottom_Face, 'Bottom_Face' )

	# Bottom_Ext - External surfaces for the bottom disk for HTC
	Bottom_Ext_ID = Bottom_Ext.GetSubShapeIndices()
	ID = SalomeFunc.ObjIndex(Testpiece, Cylinder_b_orig, Bottom_Ext_ID)[0]
	Bottom_Ext = geompy.CreateGroup(Testpiece, geompy.ShapeType["FACE"])
	geompy.UnionIDs(Bottom_Ext, ID)
	geompy.addToStudyInFather( Testpiece, Bottom_Ext, 'Bottom_Ext' )

	if isVoid:
		# On void created previously the surfaces are made up of from the
		# geometric IDs [3,10,12]
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

	globals().update(locals())

	return Mesh_1

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
		Create(TestDimensions())
	# 1 argument provided which is the parameter file
	elif len(sys.argv) == 2:
		ParameterFile = sys.argv[1]
		sys.path.insert(0, os.path.dirname(ParameterFile))
		Parameters = __import__(os.path.splitext(os.path.basename(ParameterFile))[0])
		Create(Parameters)
