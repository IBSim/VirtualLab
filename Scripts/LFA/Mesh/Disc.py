import sys
sys.dont_write_bytecode=True
import numpy as np
import os
from types import SimpleNamespace
from Scripts.Common.VLFunctions import VerifyParameters

'''
This script generates a sample using the SALOME software package which is used
to perform a Laser Flash Analysis (LFA) simulation. An optional void can be
included in the sample. An image of the sample and reference to variable names
can be found in:
https://gitlab.com/ibsim/media/-/blob/master/images/VirtualLab/LFA_Disc.png

The function 'Example' contains the variables and values required to generate a
mesh.
'''

def Example():
	Parameters = SimpleNamespace(Name='Test')
	# === Geometrical dimensions ===
	Parameters.Radius = 0.0063
	Parameters.HeightB = 0.00125
	Parameters.HeightT = 0.00125
	Parameters.VoidCentre = (0,0)
	Parameters.VoidRadius = 0.0005
	Parameters.VoidHeight = -0.0001
	### Mesh parameters
	Parameters.Length1D = 0.0004 # Length on  edges
	Parameters.Length2D = 0.0004 # Maximum length of any edge belonging to a face
	Parameters.Length3D = 0.0004 # Maximum length of any edge belogining to a tetrahedron
	Parameters.VoidSegmentN = 20

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
	ReqVar = ['Radius','HeightB','HeightT',
			  'Length1D','Length2D','Length3D']
	# Optional variables - all are needed to create a void
	OptVar = ['VoidCentre','VoidRadius','VoidHeight','VoidSegmentN']

	miss = VerifyParameters(Parameters,ReqVar)
	if miss:
		error.append("The following variables are not declared in the "\
					 "mesh parameters:\n{}".format("\n".join(miss)))

	miss = VerifyParameters(Parameters,OptVar)
	if miss and len(miss)<len(OptVar):
		error.append("Following variables are not declared in the "\
					 "mesh parameters:\n{}".format("\n".join(miss)))
	if error:
		return error, warning

	# =============================================================
	# Check conditons based on the dimensions provided
	if hasattr(Parameters,'VoidHeight'):
		if Parameters.VoidHeight >= Parameters.HeightT:
			error.append('VoidHeight is greater than HeightT')
		elif -1*Parameters.VoidHeight >= Parameters.HeightB:
			error.append('VoidHeight is greater than HeightB')
		if Parameters.VoidRadius >= Parameters.Radius:
			error.append('VoidRadius is greater than Radius')
		if (Parameters.VoidHeight == 0 and  Parameters.VoidRadius !=0) or (Parameters.VoidHeight != 0 and  Parameters.VoidRadius ==0):
			error.append('VoidHeight and VoidRadius must both be zero or non-zero')

	return error, warning

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

	isVoid = True if Parameters.VoidHeight else False

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

	### Bottom disc
	Disc_b_orig = geompy.MakeCylinder(O, OZ, Parameters.Radius, Parameters.HeightB)
	geompy.addToStudy(Disc_b_orig,'Disc_b_orig')

	## Top disc
	Vertex_1 = geompy.MakeVertexWithRef(O, 0, 0, Parameters.HeightB)
	Disc_t_orig = geompy.MakeCylinder(Vertex_1, OZ, Parameters.Radius, Parameters.HeightT)
	geompy.addToStudy(Disc_t_orig,'Disc_t_orig')

	if isVoid:
		Vertex_Void = geompy.MakeVertexWithRef(O, *Parameters.VoidCentre, Parameters.HeightB)
		VoidBase = geompy.MakeDiskPntVecR(Vertex_Void, OZ, Parameters.VoidRadius)
		Void = geompy.MakePrismVecH(VoidBase, OZ, Parameters.VoidHeight)
		Disc_t = geompy.MakeCutList(Disc_t_orig,[Void], True)
		Disc_b = geompy.MakeCutList(Disc_b_orig,[Void], True)

		# joining face - need the newly created face due to void
		_shface = geompy.GetSubShape(Disc_b_orig,[10]) # top face of the bottom disc
		_JoinFace = geompy.MakeCutList(_shface,[VoidBase], True)
	else:
		Disc_t = Disc_t_orig
		Disc_b = Disc_b_orig

		# joining face
		_JoinFace = geompy.GetSubShape(Disc_b_orig,[10])

	Disc_Fuse = geompy.MakeFuseList([Disc_b, Disc_t], True, True)
	Testpiece = geompy.MakePartition([Disc_Fuse], [Disc_b], [], [], geompy.ShapeType["SOLID"], 0, [], 0)

	geompy.addToStudy(Testpiece, 'Testpiece')

	## Add Groups
	# Volumes
	TopID = geompy.GetSameIDs(Testpiece,Disc_t)
	Top = SalomeFunc.AddGroup(Testpiece,'Top',TopID)

	BottomID = geompy.GetSameIDs(Testpiece,Disc_b)
	Bottom = SalomeFunc.AddGroup(Testpiece,'Bottom',BottomID)

	# Faces
	JoinID = geompy.GetSameIDs(Testpiece,_JoinFace)
	Join_Face = SalomeFunc.AddGroup(Testpiece,'Join_Face',JoinID)

	BottomID = SalomeFunc.ObjIndex(Testpiece, Disc_b_orig, [12])[0]
	Bottom_Face = SalomeFunc.AddGroup(Testpiece,'Bottom_Face',BottomID)

	TopID = SalomeFunc.ObjIndex(Testpiece, Disc_t_orig, [10])[0]
	Top_Face = SalomeFunc.AddGroup(Testpiece,'Top_Face',TopID)

	ExtID = geompy.SubShapeAllIDs(Testpiece,geompy.ShapeType['FACE'])
	External_Faces = SalomeFunc.AddGroup(Testpiece,'External_Faces',ExtID)

	BottomExtID = SalomeFunc.ObjIndex(Testpiece, Disc_b_orig, [3,12])[0]
	Bottom_Ext = SalomeFunc.AddGroup(Testpiece,'Bottom_Ext',BottomExtID)

	TopExtID = SalomeFunc.ObjIndex(Testpiece, Disc_t_orig, [3,10])[0]
	Top_Ext = SalomeFunc.AddGroup(Testpiece,'Top_Ext',TopExtID)

	if isVoid:
		VoidID = SalomeFunc.ObjIndex(Testpiece, Void, [3, 10, 12])[0]
		Void_Ext = SalomeFunc.AddGroup(Testpiece,'Void_Ext',VoidID)


	###
	### SMESH component
	###

	### Create Main Mesh
	Mesh_1 = smesh.Mesh(Testpiece)
	Regular_1D = Mesh_1.Segment()
	Local_Length_1 = Regular_1D.LocalLength(Parameters.Length1D,None,1e-07)
	NETGEN_2D = Mesh_1.Triangle(algo=smeshBuilder.NETGEN_2D)
	NETGEN_2D_Parameters_1 = NETGEN_2D.Parameters()
	NETGEN_2D_Parameters_1.SetMaxSize( Parameters.Length2D )
	NETGEN_2D_Parameters_1.SetOptimize( 1 )
	NETGEN_2D_Parameters_1.SetFineness( 3 )
	NETGEN_2D_Parameters_1.SetChordalError( 0.1 )
	NETGEN_2D_Parameters_1.SetChordalErrorEnabled( 0 )
	NETGEN_2D_Parameters_1.SetMinSize( Parameters.Length2D )
	NETGEN_2D_Parameters_1.SetUseSurfaceCurvature( 1 )
	NETGEN_2D_Parameters_1.SetQuadAllowed( 0 )
	NETGEN_3D = Mesh_1.Tetrahedron()
	NETGEN_3D_Parameters_1 = NETGEN_3D.Parameters()
	NETGEN_3D_Parameters_1.SetMaxSize( Parameters.Length3D )
	NETGEN_3D_Parameters_1.SetOptimize( 1 )
	NETGEN_3D_Parameters_1.SetFineness( 2 )
	NETGEN_3D_Parameters_1.SetMinSize( Parameters.Length3D )

	smesh.SetName(Regular_1D.GetAlgorithm(), 'Regular_1D')
	smesh.SetName(NETGEN_3D.GetAlgorithm(), 'NETGEN 3D')
	smesh.SetName(NETGEN_2D.GetAlgorithm(), 'NETGEN 2D')
	smesh.SetName(NETGEN_2D_Parameters_1, 'NETGEN 2D Parameters_1')
	smesh.SetName(Local_Length_1, 'Local Length_1')
	smesh.SetName(NETGEN_3D_Parameters_1, 'NETGEN 3D Parameters_1')


	## Add groups
	Mesh_Top = Mesh_1.GroupOnGeom(Top,'Top',SMESH.VOLUME)
	Mesh_Top = Mesh_1.GroupOnGeom(Bottom,'Bottom',SMESH.VOLUME)

	Mesh_Top_Ext = Mesh_1.GroupOnGeom(Top_Ext,'Top_Ext',SMESH.FACE)
	Mesh_Bottom_Ext = Mesh_1.GroupOnGeom(Bottom_Ext,'Bottom_Ext',SMESH.FACE)
	Mesh_Top_Face = Mesh_1.GroupOnGeom(Top_Face,'Top_Face',SMESH.FACE)
	Mesh_Bottom_Face = Mesh_1.GroupOnGeom(Bottom_Face,'Bottom_Face',SMESH.FACE)
	# Mesh_External = Mesh_1.GroupOnGeom(External_Faces,'External_Faces',SMESH.FACE)
	Mesh_Join_Face = Mesh_1.GroupOnGeom(Join_Face,'Contact',SMESH.FACE)

	Mesh_Bottom_Ext_Node = Mesh_1.GroupOnGeom(Bottom_Face,'NBottom_Face',SMESH.NODE)
	Mesh_Top_Ext_Node = Mesh_1.GroupOnGeom(Top_Face,'NTop_Face',SMESH.NODE)

	if isVoid:
		HoleCirc = 2*np.pi*Parameters.VoidRadius
		HoleLength = HoleCirc/Parameters.VoidSegmentN

		### Sub Mesh creation
		## Sub-Mesh 2
		Regular_1D_2 = Mesh_1.Segment(geom=Void_Ext)
		Local_Length_2 = Regular_1D_2.LocalLength(HoleLength,None,1e-07)
		NETGEN_2D_2 = Mesh_1.Triangle(algo=smeshBuilder.NETGEN_2D,geom=Void_Ext)
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

		Mesh_Void_Ext = Mesh_1.GroupOnGeom(Void_Ext,'Void_Ext',SMESH.FACE)

	isDone = Mesh_1.Compute()

	Affected = Mesh_1.AffectedElemGroupsInRegion( [ Mesh_Join_Face ], [], None )
	NewGrps = Mesh_1.DoubleNodeElemGroups( [Mesh_Join_Face], [], Affected, 1, 0 )
	[Mesh_1.RemoveGroup(grp) for grp in Affected]

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
