import os
import sys
import numpy as np
from salome.geom import geomBuilder
from salome.smesh import smeshBuilder
import SMESH
import salome_version
import SalomeFunc
import salome
import contextlib

if salome_version.getVersions()[0] < 9:
	import salome
	theStudy = salome.myStudy
	geompy = geomBuilder.New(theStudy)
	smesh = smeshBuilder.New(theStudy)
else :
	geompy = geomBuilder.New()
	smesh = smeshBuilder.New()

class GetMesh():
	def __init__(self, Mesh):

		self.Geom = Mesh.GetShape()

		self.MainMesh = {'Ix':geompy.SubShapeAllIDs(self.Geom, geompy.ShapeType["SOLID"])}
		MeshInfo = Mesh.GetHypothesisList(self.Geom)
		MeshAlgo, MeshHypoth = MeshInfo[::2], MeshInfo[1::2]
		for algo, hypoth in zip(MeshAlgo, MeshHypoth):
			self.MainMesh[algo.GetName()] = hypoth

		SubMeshes = Mesh.GetMeshOrder() if Mesh.GetMeshOrder() else Mesh.GetMesh().GetSubMeshes()
		self.SubMeshes = []
		for sm in SubMeshes:
			Geom = sm.GetSubShape()
			dict = {"Ix":Geom.GetSubShapeIndices()}

			smInfo = Mesh.GetHypothesisList(Geom)
			smAlgo, smHypoth = smInfo[::2], smInfo[1::2]
			for algo, hypoth in zip(smAlgo, smHypoth):
				dict[algo.GetName()] = hypoth

			self.SubMeshes.append(dict)

		self.Groups = {'NODE':{},'EDGE':{},'FACE':{}, 'VOLUME':{}}
		for grp in Mesh.GetGroups():
			GrpType = str(grp.GetType())
			shape = grp.GetShape()

			Ix = self.MainMesh['Ix'] if shape.IsMainShape() else shape.GetSubShapeIndices()
			Name = str(grp.GetName())

			self.Groups[GrpType][Name] = Ix



def CreateEMMesh(objMesh,Parameter):
	O = geompy.MakeVertex(0, 0, 0)
	OX = geompy.MakeVectorDXDYDZ(1, 0, 0)
	OY = geompy.MakeVectorDXDYDZ(0, 1, 0)
	OZ = geompy.MakeVectorDXDYDZ(0, 0, 1)

	### Moving the Sample to the desired location
	SampleMesh = GetMesh(objMesh)

	cPipeIn = geompy.MakeCDG(geompy.GetSubShape(SampleMesh.Geom, SampleMesh.Groups['FACE']['PipeIn']))
	cPipeOut = geompy.MakeCDG(geompy.GetSubShape(SampleMesh.Geom, SampleMesh.Groups['FACE']['PipeOut']))

	# Translate Sample so that centre of the pipe is at the origin
	CrdPipeIn = np.array(geompy.PointCoordinates(cPipeIn))
	CrdPipeOut = np.array(geompy.PointCoordinates(cPipeOut))
	Translation = -(CrdPipeIn + CrdPipeOut)/2
	Sample = geompy.MakeTranslation(SampleMesh.Geom, Translation[0], Translation[1], Translation[2])

	# Orientate Sample so that the pipe runs along the Y axis
	PipeVect = geompy.MakeVector(cPipeIn, cPipeOut)
	Rotate1 = geompy.GetAngleRadians(PipeVect, OY)
	Sample = geompy.MakeRotation(Sample, OZ, -Rotate1)

	# Orientate Sample so that the norm to CoilFace is in the Z axis
	CoilNorm = geompy.GetNormal(geompy.GetSubShape(SampleMesh.Geom, SampleMesh.Groups['FACE']['CoilFace']))
	Rotate2 = geompy.GetAngleRadians(CoilNorm, OZ)
	Sample = geompy.MakeRotation(Sample, OY, -Rotate2)

	### Importing coil design and moving to the desired location
	# CoilDict = Parameter.Coil
	from EM import CoilDesigns
	CoilFnc = getattr(CoilDesigns, Parameter.CoilType)
	CoilMesh = GetMesh(CoilFnc())

	cCoilIn = geompy.MakeCDG(geompy.GetSubShape(CoilMesh.Geom, CoilMesh.Groups['FACE']['CoilIn']))
	cCoilOut = geompy.MakeCDG(geompy.GetSubShape(CoilMesh.Geom, CoilMesh.Groups['FACE']['CoilOut']))
	CoilVect = geompy.MakeVector(cCoilIn, cCoilOut)
	CrdCoilIn = np.array(geompy.PointCoordinates(cCoilIn))
	CrdCoilOut = np.array(geompy.PointCoordinates(cCoilOut))
	CrdCoilMid = (CrdCoilIn + CrdCoilOut)/2

	# Find point CoilTight where the Coil BoundingBox sits on top of the Sample BoundingBox
	SampleZmax = geompy.BoundingBox(Sample)[5]

	CoilZmin = geompy.BoundingBox(geompy.MakeBoundingBox(CoilMesh.Geom,True))[4]
	CoilTight = [0.090915, 0, SampleZmax + (CrdCoilMid[2] - CoilZmin)]
	CoilTerminal = np.array(CoilTight) + np.array(Parameter.CoilDisplacement)

	Translation = np.array(CoilTerminal) - CrdCoilMid
	Coil = geompy.MakeTranslation(CoilMesh.Geom, Translation[0], Translation[1], Translation[2])

	RotateVector = geompy.MakeTranslation(OZ, CoilTerminal[0], CoilTerminal[1], CoilTerminal[2])
	RotateAngle = geompy.GetAngleRadians(CoilVect, OY)
	Coil = geompy.MakeRotation(Coil, RotateVector, -RotateAngle)

	Sample = geompy.MakeRotation(Sample, OY, Parameter.CoilRotation/180*np.pi)

	Common = geompy.MakeCommonList([Sample,Coil], True)
	Measure = np.array(geompy.BasicProperties(Common))
	Common.Destroy()
	if not all(Measure < 1e-9):
		return 2319

	### Creating Chamber consisting of the sample, coil and vacuum
	Compound = geompy.MakeCompound([Sample, Coil])
	[Xmin,Xmax,Ymin,Ymax,Zmin,Zmax] = geompy.BoundingBox(Compound)

	# Create sphere for vacuum
	VacuumScale = 2
	# Vacuum centre at centre of Coil/Sample compound
	VacuumCentre = geompy.MakeVertex(0.5*(Xmin + Xmax),0.5*(Ymin + Ymax),0.5*(Zmin +Zmax))
	# Vacuum radius is the distance from VacuumCentre to the corner of the bounding box multiplied by VacuumScale
	VacuumRad = np.linalg.norm([Xmax-Xmin,Ymax-Ymin,Zmax-Zmin])*VacuumScale
	Vacuum = geompy.MakeSpherePntR(VacuumCentre, VacuumRad)

	Chamber = geompy.MakePartition([Vacuum], [Compound], [], [], geompy.ShapeType["SOLID"], 0, [], 0)
	geompy.addToStudy( Chamber, 'Chamber' )


	### Meshing part ###
	### Start by meshing the whole domain coarsly, and then add sub meshes for the sample and coil
	### This is in essence the sub mesh for the vacuum

	### Main Mesh
	# Mesh Parameters
	VacCirc = 2*np.pi*VacuumRad
	VacSeg = 25
	Vac1D = VacCirc/VacSeg
	Vac2D = Vac1D
	Vac3D = Vac1D

	ERMES = smesh.Mesh(Chamber)

	Vacuum_1D = ERMES.Segment()
	Vacuum_1D_Parameters = Vacuum_1D.LocalLength(Vac1D,None,1e-07)

	Vacuum_2D = ERMES.Triangle(algo=smeshBuilder.NETGEN_2D)
	Vacuum_2D_Parameters = Vacuum_2D.Parameters()
	Vacuum_2D_Parameters.SetOptimize( 1 )
	Vacuum_2D_Parameters.SetFineness( 3 )
	Vacuum_2D_Parameters.SetChordalError( 0.1 )
	Vacuum_2D_Parameters.SetChordalErrorEnabled( 0 )
	Vacuum_2D_Parameters.SetUseSurfaceCurvature( 1 )
	Vacuum_2D_Parameters.SetQuadAllowed( 0 )
	Vacuum_2D_Parameters.SetMaxSize( Vac2D )
	Vacuum_2D_Parameters.SetMinSize( 0.001 )

	Vacuum_3D = ERMES.Tetrahedron()
	Vacuum_3D_Parameters = Vacuum_3D.Parameters()
	Vacuum_3D_Parameters.SetOptimize( 1 )
	Vacuum_3D_Parameters.SetFineness( 3 )
	Vacuum_3D_Parameters.SetMaxSize( Vac3D )
	Vacuum_3D_Parameters.SetMinSize( 0.001 )

	smesh.SetName(Vacuum_1D_Parameters, 'Vacuum_1D_Parameters')
	smesh.SetName(Vacuum_2D_Parameters, 'Vacuum_2D_Parameters')
	smesh.SetName(Vacuum_3D_Parameters, 'Vacuum_3D_Parameters')


	# Coil Mesh parameters which will be added as a sub-mesh
	CoilOrder = []

	Ix = SalomeFunc.ObjIndex(Chamber, Coil, CoilMesh.MainMesh['Ix'], Strict=False)[0]
	Geom = geompy.GetSubShape(Chamber, Ix)

	Param1D = CoilMesh.MainMesh.get('Regular_1D', None)
	if Param1D:
		Coil1D = ERMES.Segment(geom=Geom)
		ERMES.AddHypothesis(Param1D, geom=Geom)

	Param2D = CoilMesh.MainMesh.get('NETGEN_2D_ONLY', None)
	if Param2D:
		Coil2D = ERMES.Triangle(algo=smeshBuilder.NETGEN_2D,geom=Geom)
		ERMES.AddHypothesis(Param2D, geom=Geom)

	Param3D = CoilMesh.MainMesh.get('NETGEN_3D', None)
	if Param3D:
		Coil3D = ERMES.Tetrahedron(geom=Geom)
		ERMES.AddHypothesis(Param3D, geom=Geom)


	CoilSub = Coil1D.GetSubMesh()
	CoilOrder.append(CoilSub)
	smesh.SetName(CoilSub, 'Coil')


	# Adding Sample Mesh information to Chamber. Add sub meshes first as these take priority over
	# the main mesh in the ordering
	SampleOrder = []
	Mesh_Sample = SampleMesh.SubMeshes + [SampleMesh.MainMesh]
	for i, sm in enumerate(Mesh_Sample):
		Ix = SalomeFunc.ObjIndex(Chamber, Sample, sm['Ix'])[0]
		Geom = geompy.GetSubShape(Chamber, Ix)
		Param1D = sm.get('Regular_1D', None)
		if Param1D:
			mesh1D = ERMES.Segment(geom=Geom)
			ERMES.AddHypothesis(Param1D, geom=Geom)

		Param2D = sm.get('NETGEN_2D_ONLY', None)
		if Param2D:
			mesh2D = ERMES.Triangle(algo=smeshBuilder.NETGEN_2D,geom=Geom)
			ERMES.AddHypothesis(Param2D, geom=Geom)

		Param3D = sm.get('NETGEN_3D', None)
		if Param3D:
			mesh3D = ERMES.Tetrahedron(geom=Geom)
			ERMES.AddHypothesis(Param3D, geom=Geom)

		submesh = mesh1D.GetSubMesh()
		SampleOrder.append(submesh)
		smesh.SetName(submesh, "Sample_{}".format(i+1))

	# Prioritise the Coil over the Sample (shouldn't matter as these are seperate geometries)
	SubMeshOrd = CoilOrder + SampleOrder
	ERMES.SetMeshOrder([SubMeshOrd])


	### Add Groups ###
	# Groups for Sample
	SampleGrps, SampleNames = [], []
	NodesName, NodesIx = zip(*SampleMesh.Groups['NODE'].items())
	NodesName, NodesIx = list(NodesName), list(NodesIx)

	# Creating element groups (and their nodal counterparts if they exist)
	for grptype in ['EDGE','FACE','VOLUME']:
		for Name, Ix in SampleMesh.Groups[grptype].items():
			NewIx = SalomeFunc.ObjIndex(Chamber, Sample, list(Ix))[0]
			grp = SalomeFunc.AddGroup(Chamber, Name, NewIx)
			meshgrp = ERMES.GroupOnGeom(grp, Name, getattr(SMESH, grptype))

			SampleGrps.append(meshgrp)
			if Ix in NodesIx:
				listindex = NodesIx.index(Ix)
				NdName = NodesName.pop(listindex)
				NodesIx.pop(listindex)
			else:	NdName = None

			SampleNames += [NdName,Name]
	# Creating nodal groups which are not related to an element group
	# Needs sorting
#	for Name, Ix in zip(NodesName,NodesIx):
#	print(NodesName,NodesIx)



	# Groups for Coil
	EMGrps, EMNames = [], []
	for grptype, grpdict in CoilMesh.Groups.items():
		for Name, Ix in grpdict.items():
			NewIx = SalomeFunc.ObjIndex(Chamber, Coil, Ix,Strict=False)[0]
			grp = SalomeFunc.AddGroup(Chamber, Name, NewIx)
			meshgrp = ERMES.GroupOnGeom(grp, Name, getattr(SMESH, grptype))
			EMGrps.append(meshgrp)
			EMNames += [None, Name]

	# Groups for Vacuum
	grp = SalomeFunc.AddGroup(Chamber, 'Vacuum', [2])
	meshgrp = ERMES.GroupOnGeom(grp, 'Vacuum', SMESH.VOLUME)
	EMGrps.append(meshgrp)
	EMNames += [None, 'Vacuum']

	# Vacuum External Surface
	Ix = SalomeFunc.ObjIndex(Chamber, Vacuum, [3])[0]
	grp = SalomeFunc.AddGroup(Chamber, 'VacuumSurface', Ix)
	meshgrp = ERMES.GroupOnGeom(grp, 'VacuumSurface', SMESH.FACE)
	EMGrps.append(meshgrp)
	EMNames += [None, 'VacuumSurface']

	EMGrps = SampleGrps + EMGrps
	EMNames = SampleNames + EMNames
	# Remove Node groups from the EMMesh as we dont need them
	EMNames[0::2] = [None]*len(EMGrps)

	print('Computing ERMES mesh')
	isdone = ERMES.Compute()

	SampleMesh = smesh.Concatenate(SampleGrps, 1, 1, 1e-05,True,'Sample')
	for name, grp in zip(SampleNames, SampleMesh.GetGroups()):
		if name: grp.SetName(name)
		else : SampleMesh.RemoveGroup(grp)

	ERMESMesh = smesh.Concatenate(EMGrps, 1, 1, 1e-05,True,'xERMES')
	for name, grp in zip(EMNames, ERMESMesh.GetGroups()):
		if name: grp.SetName(name)
		else : ERMESMesh.RemoveGroup(grp)

	with open("/dev/null", 'w') as f:
		with contextlib.redirect_stdout(f):
			SampleMesh.Compute()
			ERMESMesh.Compute()

	# b = salome.myStudy.NewBuilder()
	# so =  salome.ObjectToSObject(ERMES.mesh)
	# b.RemoveObjectWithChildren(so)

	return SampleMesh, ERMESMesh, locals()


def testgeom():
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

			self.TileCentre = [0,-0.01]
			self.TileWidth = self.BlockWidth
			self.TileLength = 0.03
			self.TileHeight = 0.005

			### Mesh
			self.Length1D = 0.005
			self.Length2D = 0.005
			self.Length3D = 0.005
			self.CircDisc = 20
			self.Sub2_1D = 0.003

			self.CoilType = 'HIVE'
			self.CoilDisp = [0, 0.005, 0.005]

			self.MeshName = 'Test'
			self.SampleGroups = ['Tile','Block','Pipe']

	Parameter = TestDimensions()

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

	## Creating the sample
	# Pipe
	InRad = Parameter.PipeDiam/2
	OutRad = Parameter.PipeDiam/2 + Parameter.PipeThick

	Vertex_1 = geompy.MakeVertex(Parameter.BlockWidth/2+Parameter.PipeCentre[0], -0.5*(Parameter.PipeLength-Parameter.BlockLength), Parameter.BlockHeight/2+Parameter.PipeCentre[1])
	Fluid = geompy.MakeCylinder(Vertex_1, OY, InRad, Parameter.PipeLength)
	PipeExt = geompy.MakeCylinder(Vertex_1, OY, OutRad, Parameter.PipeLength)
	Pipe = geompy.MakeCutList(PipeExt, [Fluid], True)

	geompy.addToStudy( Pipe, 'Pipe' )

	# Block
	Box = geompy.MakeBoxDXDYDZ(Parameter.BlockWidth, Parameter.BlockLength, Parameter.BlockHeight)
	Block = geompy.MakeCutList(Box, [PipeExt], True)
	geompy.addToStudy( Block, 'Block')

	# Tile
	TileCentre = geompy.MakeVertex(Parameter.BlockWidth/2+Parameter.TileCentre[0], Parameter.BlockLength/2+Parameter.TileCentre[1], Parameter.BlockHeight)
	TileCorner1 = geompy.MakeVertexWithRef(TileCentre, -Parameter.TileWidth/2, -Parameter.TileLength/2, 0)
	TileCorner2 = geompy.MakeVertexWithRef(TileCentre, Parameter.TileWidth/2, Parameter.TileLength/2, Parameter.TileHeight)
	Tile = geompy.MakeBoxTwoPnt(TileCorner1, TileCorner2)

	geompy.addToStudy( Tile, 'Tile')

	Fuse = geompy.MakeFuseList([Pipe, Block, Tile], True, True)
	Sample = geompy.MakePartition([Fuse], [Pipe, Block, Tile], [], [], geompy.ShapeType["SOLID"], 0, [], 0)

	geompy.addToStudy( Sample, 'Sample')

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
	PipeExtIx = SalomeFunc.ObjIndex(Sample, Pipe, [10,15,20])[0]
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
	Length1 = Parameter.PipeDiam*np.pi/Parameter.CircDisc

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
	Local_Length_2 = Regular_1D_2.LocalLength(Parameter.Sub2_1D,None,1e-07)
	NETGEN_2D_2 = Mesh_1.Triangle(algo=smeshBuilder.NETGEN_2D,geom=GrpTile)
	NETGEN_2D_Parameters_2 = NETGEN_2D_2.Parameters()
	NETGEN_2D_Parameters_2.SetOptimize( 1 )
	NETGEN_2D_Parameters_2.SetFineness( 3 )
	NETGEN_2D_Parameters_2.SetChordalError( 0.1 )
	NETGEN_2D_Parameters_2.SetChordalErrorEnabled( 0 )
	NETGEN_2D_Parameters_2.SetUseSurfaceCurvature( 1 )
	NETGEN_2D_Parameters_2.SetQuadAllowed( 0 )
	NETGEN_2D_Parameters_2.SetMaxSize( Parameter.Sub2_1D )
	NETGEN_2D_Parameters_2.SetMinSize( Parameter.Sub2_1D )

	NETGEN_3D_2 = Mesh_1.Tetrahedron(geom=GrpTile)
	NETGEN_3D_Parameters_2 = NETGEN_3D_2.Parameters()
	NETGEN_3D_Parameters_2.SetOptimize( 1 )
	NETGEN_3D_Parameters_2.SetFineness( 3 )
	NETGEN_3D_Parameters_2.SetMaxSize( Parameter.Sub2_1D )
	NETGEN_3D_Parameters_2.SetMinSize( Parameter.Sub2_1D )

	smesh.SetName(Sub_mesh_2, 'Sub-mesh_2')
	smesh.SetName(Local_Length_2, 'Local Length_2')
	smesh.SetName(NETGEN_2D_Parameters_2, 'NETGEN 2D Parameters_2')
	smesh.SetName(NETGEN_3D_Parameters_2, 'NETGEN 3D Parameters_2')

	CreateEMMesh(Mesh_1, Parameter)

	globals().update(locals())

if __name__ == "__main__":
	sys.path.insert(0, '/home/rhydian/Documents/Scripts/Simulation/virtuallab/Scripts/HIVE')
	testgeom()
