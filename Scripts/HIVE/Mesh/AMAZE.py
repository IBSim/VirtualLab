import sys
import numpy as np
import os
sys.dont_write_bytecode=True
import time


def Create(**kwargs):

	from salome.geom import geomBuilder
	from salome.smesh import smeshBuilder
	import  SMESH
	import salome_version
	import SalomeFunc
	import salome

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

	# # flag dictating whether GUI is open
	# GUIopen = salome.sg.hasDesktop()

	O = geompy.MakeVertex(0, 0, 0)
	OX = geompy.MakeVectorDXDYDZ(1, 0, 0)
	OY = geompy.MakeVectorDXDYDZ(0, 1, 0)
	OZ = geompy.MakeVectorDXDYDZ(0, 0, 1)

	geompy.addToStudy( O, 'O' )
	geompy.addToStudy( OX, 'OX' )
	geompy.addToStudy( OY, 'OY' )
	geompy.addToStudy( OZ, 'OZ' )

	### Creating the sample geometry
	## Pipe
	InRad = Parameter.PipeDiam/2
	OutRad = Parameter.PipeDiam/2 + Parameter.PipeThick

	Vertex_1 = geompy.MakeVertex(Parameter.BlockWidth/2+Parameter.PipeCentre[0], -0.5*(Parameter.PipeLength-Parameter.BlockLength), Parameter.BlockHeight/2+Parameter.PipeCentre[1])
	Fluid = geompy.MakeCylinder(Vertex_1, OY, InRad, Parameter.PipeLength)
	PipeExt = geompy.MakeCylinder(Vertex_1, OY, OutRad, Parameter.PipeLength)
	Pipe = geompy.MakeCutList(PipeExt, [Fluid], True)
	geompy.addToStudy( Pipe, 'Pipe' )

	## Block
	Box = geompy.MakeBoxDXDYDZ(Parameter.BlockWidth, Parameter.BlockLength, Parameter.BlockHeight)
	Block = geompy.MakeCutList(Box, [PipeExt], True)
	geompy.addToStudy( Block, 'Block')

	## Tile
	TileCentre = geompy.MakeVertex(Parameter.BlockWidth/2+Parameter.TileCentre[0], Parameter.BlockLength/2+Parameter.TileCentre[1], Parameter.BlockHeight)
	TileCorner1 = geompy.MakeVertexWithRef(TileCentre, -Parameter.TileWidth/2, -Parameter.TileLength/2, 0)
	TileCorner2 = geompy.MakeVertexWithRef(TileCentre, Parameter.TileWidth/2, Parameter.TileLength/2, Parameter.TileHeight)
	Tile = geompy.MakeBoxTwoPnt(TileCorner1, TileCorner2)
	geompy.addToStudy( Tile, 'Tile')

	# Combine parts
	Fuse = geompy.MakeFuseList([Pipe, Block, Tile], True, True)
	Sample = geompy.MakePartition([Fuse], [Pipe, Block, Tile], [], [], geompy.ShapeType["SOLID"], 0, [], 0)
	geompy.addToStudy( Sample, 'Sample')

	# ObjIndex function returns the object index of a sub-shape in a new geometry
	# i.e. we want to know what the object index of a sub-shape of the tile is in
	# sample to create a group there.

	## Add Groups
	# Solid
	Ix = SalomeFunc.ObjIndex(Sample, Tile, [1])[0]
	GrpTile = SalomeFunc.AddGroup(Sample, 'Tile', Ix)

	Ix = SalomeFunc.ObjIndex(Sample, Pipe, [1])[0]
	GrpPipe = SalomeFunc.AddGroup(Sample, 'Pipe', Ix)

	Ix = SalomeFunc.ObjIndex(Sample, Block, [1])[0]
	GrpBlock = SalomeFunc.AddGroup(Sample, 'Block', Ix)

	# Surfaces
	Ix = SalomeFunc.ObjIndex(Sample, Tile, [33])[0]
	GrpCoilFace = SalomeFunc.AddGroup(Sample, 'CoilFace', Ix)

	Ix = SalomeFunc.ObjIndex(Sample, Pipe, [20])[0]
	GrpPipeFace = SalomeFunc.AddGroup(Sample, 'PipeFace', Ix)

	Ix = SalomeFunc.ObjIndex(Sample, Pipe, [15])[0]
	GrpPipeIn = SalomeFunc.AddGroup(Sample, 'PipeIn', Ix)

	Ix = SalomeFunc.ObjIndex(Sample, Pipe, [10])[0]
	GrpPipeOut = SalomeFunc.AddGroup(Sample, 'PipeOut', Ix)

	# Create SampleSurface group by taking relevant parts of each geometry
	# Tile
	TileExtIx = SalomeFunc.ObjIndex(Sample, Tile, [3,13,23,27,33])[0]
	# Pipe
	# Get the indicies of the new faces create where the Pipe joins the Block
	CutPipBlk = geompy.MakeCutList(geompy.GetSubShape(Pipe,[3]), [geompy.GetSubShape(Block,[41])], True)
	SubIDs = geompy.SubShapeAllIDs(CutPipBlk, geompy.ShapeType["FACE"])
	NewIx = SalomeFunc.ObjIndex(Sample, CutPipBlk, SubIDs)[0] if SubIDs else []
	PipeExtIx = SalomeFunc.ObjIndex(Sample, Pipe, [10,15,20])[0] + NewIx
	# Block
	# Get the indicies of the new faces create where the Tile joins the Block
	CutBlkTl = geompy.MakeCutList(geompy.GetSubShape(Block,[23]), [geompy.GetSubShape(Tile,[31])], True)
	NewIx = SalomeFunc.ObjIndex(Sample, CutBlkTl, geompy.SubShapeAllIDs(CutBlkTl, geompy.ShapeType["FACE"]))[0]
	BlockExtIx = SalomeFunc.ObjIndex(Sample, Block, [3,13,28,36,39])[0] + NewIx

	Ix = TileExtIx + PipeExtIx + BlockExtIx
	GrpSampleSurface = SalomeFunc.AddGroup(Sample, 'SampleSurface', Ix)


	###
	### SMESH component
	###

	Mesh_1 = smesh.Mesh(Sample)
	Check = Mesh_1.Segment()
	Local_Length = Check.LocalLength(Parameter.Length1D,None,1e-07)

	NETGEN_2D = Mesh_1.Triangle(algo=smeshBuilder.NETGEN_2D)
	NETGEN_2D_Parameters = NETGEN_2D.Parameters()
	NETGEN_2D_Parameters.SetMaxSize( Parameter.Length2D )
	NETGEN_2D_Parameters.SetOptimize( 1 )
	NETGEN_2D_Parameters.SetFineness( 3 )
	NETGEN_2D_Parameters.SetChordalError( 0.1 )
	NETGEN_2D_Parameters.SetChordalErrorEnabled( 0 )
	NETGEN_2D_Parameters.SetMinSize( Parameter.Length2D )
	NETGEN_2D_Parameters.SetUseSurfaceCurvature( 1 )
	NETGEN_2D_Parameters.SetQuadAllowed( 0 )

	NETGEN_3D = Mesh_1.Tetrahedron()
	NETGEN_3D_Parameters = NETGEN_3D.Parameters()
	NETGEN_3D_Parameters.SetMaxSize( Parameter.Length3D )
	NETGEN_3D_Parameters.SetOptimize( 1 )
	NETGEN_3D_Parameters.SetFineness( 3 )
	NETGEN_3D_Parameters.SetMinSize( Parameter.Length3D )

	smesh.SetName(Local_Length, 'Local Length')
	smesh.SetName(Check.GetAlgorithm(), 'Check')
	smesh.SetName(NETGEN_2D.GetAlgorithm(), 'NETGEN 2D')
	smesh.SetName(NETGEN_2D_Parameters, 'NETGEN 2D Parameters')
	smesh.SetName(NETGEN_3D.GetAlgorithm(), 'NETGEN 3D')
	smesh.SetName(NETGEN_3D_Parameters, 'NETGEN 3D Parameters')
	smesh.SetName(Mesh_1.GetMesh(), 'Sample')

	### Add Groups
	# Volume
	MSample = Mesh_1.GroupOnGeom(Sample,'Sample',SMESH.VOLUME)
	MTile = Mesh_1.GroupOnGeom(GrpTile,'Tile',SMESH.VOLUME)
	MPipe = Mesh_1.GroupOnGeom(GrpPipe,'Pipe',SMESH.VOLUME)
	MBlock = Mesh_1.GroupOnGeom(GrpBlock,'Block',SMESH.VOLUME)

	# Face
	MCoilFace = Mesh_1.GroupOnGeom(GrpCoilFace,'CoilFace',SMESH.FACE)
	MPipeFace = Mesh_1.GroupOnGeom(GrpPipeFace,'PipeFace',SMESH.FACE)
	MSampleSurface = Mesh_1.GroupOnGeom(GrpSampleSurface,'SampleSurface',SMESH.FACE)
	MPipeIn = Mesh_1.GroupOnGeom(GrpPipeIn,'PipeIn',SMESH.FACE)
	MPipeOut = Mesh_1.GroupOnGeom(GrpPipeOut,'PipeOut',SMESH.FACE)

	# Node
	MPipe = Mesh_1.GroupOnGeom(GrpPipe,'PipeNd',SMESH.NODE)
	MSample = Mesh_1.GroupOnGeom(GrpBlock,'BlockNd',SMESH.NODE)

	### Sub-Mesh 1 - Refinement on pipe
	## PipeEdges
	Length1 = Parameter.PipeDiam*np.pi/Parameter.PipeDisc

	Regular_1D_1 = Mesh_1.Segment(geom=GrpPipe)
	Sub_mesh_1 = Regular_1D_1.GetSubMesh()
	Local_Length_1 = Regular_1D_1.LocalLength(Length1,None,1e-07)
	NETGEN_2D_1 = Mesh_1.Triangle(algo=smeshBuilder.NETGEN_2D,geom=GrpPipe)
	NETGEN_2D_Parameters_1 = NETGEN_2D_1.Parameters()
	NETGEN_2D_Parameters_1.SetMaxSize( Length1 )
	NETGEN_2D_Parameters_1.SetOptimize( 1 )
	NETGEN_2D_Parameters_1.SetFineness( 3 )
	NETGEN_2D_Parameters_1.SetChordalError( 0.1 )
	NETGEN_2D_Parameters_1.SetChordalErrorEnabled( 0 )
	NETGEN_2D_Parameters_1.SetMinSize( Length1 )
	NETGEN_2D_Parameters_1.SetUseSurfaceCurvature( 1 )
	NETGEN_2D_Parameters_1.SetQuadAllowed( 0 )
	NETGEN_3D_1 = Mesh_1.Tetrahedron(geom=GrpPipe)
	NETGEN_3D_Parameters_1 = NETGEN_3D_1.Parameters()
	NETGEN_3D_Parameters_1.SetMaxSize( Length1 )
	NETGEN_3D_Parameters_1.SetOptimize( 1 )
	NETGEN_3D_Parameters_1.SetFineness( 3 )
	NETGEN_3D_Parameters_1.SetMinSize( Length1 )

	smesh.SetName(Sub_mesh_1, 'Sub-mesh_1')
	smesh.SetName(Local_Length_1, 'Local Length_1')
	smesh.SetName(NETGEN_2D_Parameters_1, 'NETGEN 2D Parameters_1')
	smesh.SetName(NETGEN_3D_Parameters_1, 'NETGEN 3D Parameters_1')

	# Update Mesh parameteres with now minimum sizes
	NETGEN_2D_Parameters.SetMinSize( Length1 )
	NETGEN_3D_Parameters.SetMinSize( Length1 )

	## Sub-Mesh 2 - Refinement on Tile
	Regular_1D_2 = Mesh_1.Segment(geom=GrpTile)
	Sub_mesh_2 = Regular_1D_2.GetSubMesh()
	Local_Length_2 = Regular_1D_2.LocalLength(Parameter.SubTile,None,1e-07)
	NETGEN_2D_2 = Mesh_1.Triangle(algo=smeshBuilder.NETGEN_2D,geom=GrpTile)
	NETGEN_2D_Parameters_2 = NETGEN_2D_2.Parameters()
	NETGEN_2D_Parameters_2.SetOptimize( 1 )
	NETGEN_2D_Parameters_2.SetFineness( 3 )
	NETGEN_2D_Parameters_2.SetChordalError( 0.1 )
	NETGEN_2D_Parameters_2.SetChordalErrorEnabled( 0 )
	NETGEN_2D_Parameters_2.SetUseSurfaceCurvature( 1 )
	NETGEN_2D_Parameters_2.SetQuadAllowed( 0 )
	NETGEN_2D_Parameters_2.SetMaxSize( Parameter.SubTile )
	NETGEN_2D_Parameters_2.SetMinSize( Parameter.SubTile )

	NETGEN_3D_2 = Mesh_1.Tetrahedron(geom=GrpTile)
	NETGEN_3D_Parameters_2 = NETGEN_3D_2.Parameters()
	NETGEN_3D_Parameters_2.SetOptimize( 1 )
	NETGEN_3D_Parameters_2.SetFineness( 3 )
	NETGEN_3D_Parameters_2.SetMaxSize( Parameter.SubTile )
	NETGEN_3D_Parameters_2.SetMinSize( Parameter.SubTile )

	smesh.SetName(Sub_mesh_2, 'Sub-mesh_2')
	smesh.SetName(Local_Length_2, 'Local Length_2')
	smesh.SetName(NETGEN_2D_Parameters_2, 'NETGEN 2D Parameters_2')
	smesh.SetName(NETGEN_3D_Parameters_2, 'NETGEN 3D Parameters_2')

	MeshERMES = getattr(Parameter,'ERMES',True)
	if MeshERMES :
		## This next part takes this mesh and adds in a coil and vacuum around it and then runs the  code.
		print('Using sample mesh information to create mesh for ERMES')
		from EM.EMChamber import CreateEMMesh
		SampleMesh, ERMESMesh, loc = CreateEMMesh(Mesh_1, Parameter)
		smesh.SetName(SampleMesh, 'Sample')
		smesh.SetName(ERMESMesh, 'xERMES')
		if MeshFile:
			SalomeFunc.MeshExport(SampleMesh,MeshFile)
			SalomeFunc.MeshExport(ERMESMesh,MeshFile, Overwrite = 0)
	else:
		Mesh_1.Compute()
		SalomeFunc.MeshExport(Mesh_1,MeshFile)

	# b = salome.myStudy.NewBuilder()
	# for i, ms in enumerate([Mesh_1,SampleMesh,ERMESMesh]):
	# 	so =  salome.ObjectToSObject(ms.mesh)
	# 	b.RemoveObjectWithChildren(so)

	globals().update(locals()) ### This adds all variables created in this function

	# return 12

class TestDimensions():
	def __init__(self):
		### Geom
		self.BlockWidth = 0.03
		self.BlockLength = 0.05
		self.BlockHeight = 0.02

		self.PipeShape = 'smooth tube'
		self.PipeCentre = [0,0]
		self.PipeDiam = 0.01 ###Inner Diameter
		self.PipeThick = 0.001
		self.PipeLength = self.BlockLength

		self.TileCentre = [0,0]
		self.TileWidth = self.BlockWidth
		self.TileLength = 0.03
		self.TileHeight = 0.005

		### Mesh
		self.Length1D = 0.005
		self.Length2D = 0.005
		self.Length3D = 0.005
		self.PipeDisc = 20
		self.SubTile = 0.003

		self.CoilType = 'Test'
		self.CoilDisp = [0, 0, 0.005]

		self.MeshName = 'Test'
		self.SampleGroups = ['Tile','Block','Pipe']


def error(Parameters):
	''' This function is imported in during the Setup to pick up any errors which will occur for the given geometrical dimension. i.e. impossible dimensions '''
	# Ensure that pipe length is always greater to or equal to block length
	message = None
	return message

def ErrorHandling(Info, ReturnCode):
	print(ReturnCode)

if __name__ == '__main__':
	if len(sys.argv) == 1:
		Create(Parameter = TestDimensions(),MeshFile = None)
	# 1 argument provided which is the parameter file
	elif len(sys.argv) == 2:
		ParameterFile = sys.argv[1]
		sys.path.insert(0, os.path.dirname(ParameterFile))
		Parameters = __import__(os.path.splitext(os.path.basename(ParameterFile))[0])
		Create(Parameter = Parameters,MeshFile = None)
