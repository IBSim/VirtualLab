import sys
import numpy as np
import os
sys.dont_write_bytecode=True
import time


def Create(Parameter):

	from salome.geom import geomBuilder
	from salome.smesh import smeshBuilder
	import  SMESH
	import salome_version
	from Scripts.Common.VLPackages.Salome import SalomeFunc
	import salome
    
	if salome_version.getVersions()[0] < 9:
		theStudy = salome.myStudy
		geompy = geomBuilder.New(theStudy)
		smesh = smeshBuilder.New(theStudy)
	else :
		geompy = geomBuilder.New()
		smesh = smeshBuilder.New()
# =============================================================================
## if there is one or multiple artificial defect..
	isVoid, LargestVoidRad, SmallestVoidRad  = [], [], []
	ScalingFactor, ScalingFlag   = [], []

	for i in range(len(Parameter.Void)): 
		if Parameter.Void[i][0] and Parameter.Void[i][1] and Parameter.Void[i][2]: 
			isVoid.append(True) 
			if Parameter.Void[i][0] > Parameter.Void[i][1]: # scale the void in y-axis
				LargestVoidRad.append(Parameter.Void[i][0]) # elliptical cylinder void
				SmallestVoidRad.append(Parameter.Void[i][1])
				ScalingFactor.append((Parameter.Void[i][1]/Parameter.Void[i][0]))
				ScalingFlag.append('AxisY')
			elif Parameter.Void[i][1] > Parameter.Void[i][0]: # scale the void in x-axis
				LargestVoidRad.append(Parameter.Void[i][1]) # elliptical cylinder void
				SmallestVoidRad.append(Parameter.Void[i][0])
				ScalingFactor.append((Parameter.Void[i][0]/Parameter.Void[i][1]))
				ScalingFlag.append('AxisX')
			else: # circular void; no scaling!
				LargestVoidRad.append(Parameter.Void[i][0])
				SmallestVoidRad.append(Parameter.Void[i][0])
				ScalingFactor.append(None)
				ScalingFlag.append(None)
		else:
			isVoid.append(False)
# =============================================================================    
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
	Tile_orig = geompy.MakeBoxTwoPnt(TileCorner1, TileCorner2)
	geompy.addToStudy( Tile_orig, 'Tile_orig')

# =============================================================================
## if there is one or multiple artificial defect, add here.
    ##MultiVoid Case
	Void_List=[] # void objects

	for i in range (len (isVoid)):
		if isVoid[i]:
        	# Void is at the top half of the tile 
			VoidCentreX = Parameter.TileWidth*Parameter.VoidCentre[i][0]
			VoidCentreY = (Parameter.BlockLength - Parameter.TileLength)*0.5 + Parameter.TileLength*Parameter.VoidCentre[i][1]
			Vertex_2 = geompy.MakeVertexWithRef(O, VoidCentreX, VoidCentreY, Parameter.BlockHeight) # extrusion vector
			Void_Temp = geompy.MakeCylinder(Vertex_2, OZ, LargestVoidRad[i], Parameter.Void[i][2])
            	# if void has elliptical base, then...
			if ScalingFlag[i] == 'AxisY':
				Void_Scaled = geompy.MakeScaleAlongAxes(Void_Temp, Vertex_2, 1, ScalingFactor[i], 1)

			elif ScalingFlag[i] == 'AxisX':
				Void_Scaled = geompy.MakeScaleAlongAxes(Void_Temp, Vertex_2, ScalingFactor[i], 1, 1)

			else:  # if void has circular base
				Void = Void_Temp

			# elliptic void and rotation with respect to the void centre
			if Parameter.Void[i][3] != 0.0 and ScalingFlag[i] != None: 
				Point_1 = geompy.MakeVertex(VoidCentreX, VoidCentreY, 0)
				Point_2 = geompy.MakeVertex(VoidCentreX, VoidCentreY, Parameter.BlockHeight)
				VectorRotation = geompy.MakeVector(Point_1, Point_2)
				Void = geompy.Rotate(Void_Scaled, VectorRotation, Parameter.Void[i][3]*np.pi/180.0)

			# no rotation but elliptic void
			if Parameter.Void[i][3] == 0 and ScalingFlag[i] != None: 
				Void = Void_Scaled

			Void_List.append(Void)

	if len (isVoid) != 0:   # void(s) case
		Tile = geompy.MakeCutList(Tile_orig, Void_List, True)  
		
	else:   # no void case
		Tile = Tile_orig 
		
# =============================================================================
## Merge the parts
	Fuse = geompy.MakeFuseList([Pipe, Block, Tile], True, True)
	Sample = geompy.MakePartition([Fuse], [Pipe, Block, Tile], [], [], geompy.ShapeType["SOLID"], 0, [], 0)

	Void_Name = [] # void geometry list (solid bodies)

	for j in range (len (isVoid)): 
		Void_Name.append('Void_' + str (j))
		geompy.addToStudy(Void_List[j], Void_Name[j])

	geompy.addToStudy(Tile,'Tile')
	geompy.addToStudy(Sample, 'Sample')
# =============================================================================

# ObjIndex function returns the object index of a sub-shape in a new geometry
# i.e. we want to know what the object index of a sub-shape of the tile is in
# sample to create a group there.

# =============================================================================
	## Add Groups
	# Solid
	Ix = SalomeFunc.ObjIndex(Sample, Tile, [1])[0]  
	GrpTile = SalomeFunc.AddGroup(Sample, 'Tile', Ix)

	Ix = SalomeFunc.ObjIndex(Sample, Pipe, [1])[0]
	GrpPipe = SalomeFunc.AddGroup(Sample, 'Pipe', Ix)

	Ix = SalomeFunc.ObjIndex(Sample, Block, [1])[0]
	GrpBlock = SalomeFunc.AddGroup(Sample, 'Block', Ix)

	# Surfaces
	Ix = SalomeFunc.ObjIndex(Sample, Tile_orig, [33])[0]  
	GrpCoilFace = SalomeFunc.AddGroup(Sample, 'CoilFace', Ix)

	Ix = SalomeFunc.ObjIndex(Sample, Pipe, [20])[0]
	GrpPipeFace = SalomeFunc.AddGroup(Sample, 'PipeFace', Ix)

	Ix = SalomeFunc.ObjIndex(Sample, Pipe, [15])[0]
	GrpPipeIn = SalomeFunc.AddGroup(Sample, 'PipeIn', Ix)

	Ix = SalomeFunc.ObjIndex(Sample, Pipe, [10])[0]
	GrpPipeOut = SalomeFunc.AddGroup(Sample, 'PipeOut', Ix)

	# Create group SampleSurface which is the external surface of the sample
	# Take relevant parts from the Pipe, Tile and Block

	# Pipe
	PipeExtIx = SalomeFunc.ObjIndex(Sample, Pipe, [10,15,20])[0]
	# Get the index of the new faces create where the Pipe joins the Block
	# Get the faces remaining after block surface cut from pipe & their indexes
	CutPipBlk = geompy.MakeCutList(geompy.GetSubShape(Pipe,[3]), [geompy.GetSubShape(Block,[41])], True)
	_PipeIx = geompy.SubShapeAllIDs(CutPipBlk, geompy.ShapeType["FACE"])
	if _PipeIx:
		# Get corresponding index on Sample
		PipeIx = SalomeFunc.ObjIndex(Sample, CutPipBlk, _PipeIx)[0]
		PipeExtIx += PipeIx

	# Block
	BlockExtIx = SalomeFunc.ObjIndex(Sample, Block, [3,13,28,36,39])[0]
	# Get the index of the new faces create where the Tile joins the Block
	# This finds the additional faces on the block
	CutBlkTl = geompy.MakeCutList(geompy.GetSubShape(Block,[23]), [geompy.GetSubShape(Tile_orig,[31])], True)
	_BlkIx = geompy.SubShapeAllIDs(CutBlkTl, geompy.ShapeType["FACE"])
	if _BlkIx:
		BlkIx = SalomeFunc.ObjIndex(Sample, CutBlkTl, _BlkIx)[0]
		BlockExtIx += BlkIx

	# Tile
	TileExtIx = SalomeFunc.ObjIndex(Sample, Tile_orig, [3,13,23,27,33])[0]
	# Get the index of the new faces create where the Tile joins the Block
	# This finds the additional faces on the tile
	CutTlBlk = geompy.MakeCutList(geompy.GetSubShape(Tile,[31]), [geompy.GetSubShape(Block,[23])], True)
	_TlIx = geompy.SubShapeAllIDs(CutTlBlk, geompy.ShapeType["FACE"])
	if _TlIx:
		TlIx = SalomeFunc.ObjIndex(Sample, CutTlBlk, _TlIx)[0]
		TileExtIx += TlIx

	# Add group made up of Tile, Block and Pipe external surfaces
	GrpSampleSurface = SalomeFunc.AddGroup(Sample, 'SampleSurface', TileExtIx + PipeExtIx + BlockExtIx)
#============================================================================
# Store the Tile surfaces used for thermocouples in HIVE experimental tests (top and side surfaces) in a surface groups
	#if len(Parameter.ThermoCouple) > 0:
	TC_ID = [['Tile', 'Front', 13], ['Tile', 'Back', 3], ['Tile', 'SideA', 23], ['Tile', 'SideB', 27], ['Tile', 'Top', 33], ['Block', 'Front', 39], ['Block', 'Back', 3], ['Block', 'SideA', 13], ['Block', 'SideB', 28],['Block', 'Bottom', 36]] # TC_ID: ThermoCouple ID

	GrpThermocouple = []
	TCSurface = [] 
	for TC in TC_ID:
		SurfaceName = TC[0] + TC[1]	
		if TC[0] == 'Tile': SurfaceID = SalomeFunc.ObjIndex(Sample, Tile_orig, [TC[2]])[0]
		if TC[0] == 'Block': SurfaceID = SalomeFunc.ObjIndex(Sample, Block, [TC[2]])[0] 
		temp = SalomeFunc.AddGroup(Sample, SurfaceName, SurfaceID)
		GrpThermocouple.append(temp)
		TCSurface.append(SurfaceName)

#============================================================================
# Store the void surfaces for feature local meshing
	VoidSurfaces = []
	VoidSurfaces_Names = []	

	for k in range (len (isVoid)):
	#VoidSurfaces- Find the faces corresponding to 3(side), 10(top) and 12(bottom) in Void
		Ix = SalomeFunc.ObjIndex(Sample, Void_List[k], [3, 10, 12])[0]
		VoidSurfaces_obj = geompy.CreateGroup(Sample, geompy.ShapeType["FACE"])
		geompy.UnionIDs(VoidSurfaces_obj, Ix)
		VoidSurfaces_Names.append('Void_' + str(k) + '_Surfaces')
		VoidSurfaces.append(VoidSurfaces_obj)
		geompy.addToStudyInFather( Sample, VoidSurfaces[k], VoidSurfaces_Names[k] )

#============================================================================
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

	## Add Groups
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
            # Store surface mesh of tile for HIVE Temp Distribution Analysis
	MThermocouple = []
	for m in range (len(GrpThermocouple)):
		SurfaceName = TCSurface[m]
		MThermocouple.append(Mesh_1.GroupOnGeom(GrpThermocouple[m], TCSurface[m], SMESH.FACE))

	# Node
	MPipe = Mesh_1.GroupOnGeom(GrpPipe,'PipeNd',SMESH.NODE)
	MSample = Mesh_1.GroupOnGeom(GrpBlock,'BlockNd',SMESH.NODE)


	### Sub-Mesh 1 - Refinement on pipe
	## PipeEdges
	Length1 = Parameter.PipeDiam*np.pi/Parameter.PipeSegmentN

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

	#local (fine) meshing around voids in order to improve accuracy
	for m in range (len (isVoid)): 
		if isVoid[m]:
			local_mesh_size = (2.0*np.pi*SmallestVoidRad[m])/Parameter.VoidSegmentN
    		### Sub Mesh creation
    		## Sub-Mesh 3
			Regular_1D_3 = Mesh_1.Segment(geom=VoidSurfaces[m])
			Local_Length_3 = Regular_1D_3.LocalLength(local_mesh_size,None,1e-07)
			NETGEN_2D_3 = Mesh_1.Triangle(algo=smeshBuilder.NETGEN_2D,geom=VoidSurfaces[m])
			NETGEN_2D_Parameters_3 = NETGEN_2D_3.Parameters()
			NETGEN_2D_Parameters_3.SetMaxSize( local_mesh_size )
			NETGEN_2D_Parameters_3.SetOptimize( 1 )
			NETGEN_2D_Parameters_3.SetFineness( 3 )
			NETGEN_2D_Parameters_3.SetChordalError( 0.1 )
			NETGEN_2D_Parameters_3.SetChordalErrorEnabled( 0 )
			NETGEN_2D_Parameters_3.SetMinSize( local_mesh_size )
			NETGEN_2D_Parameters_3.SetUseSurfaceCurvature( 1 )
			NETGEN_2D_Parameters_3.SetQuadAllowed( 0 )
			Sub_mesh_3 = NETGEN_2D_3.GetSubMesh()

			smesh.SetName(Sub_mesh_3, 'Void_' + str(m) + '_Mesh')
			smesh.SetName(NETGEN_2D_Parameters_3, 'NETGEN 2D Parameters-Void_' + str(m))

			NETGEN_2D_Parameters_1.SetMinSize( local_mesh_size )
			NETGEN_3D_Parameters_1.SetMinSize( local_mesh_size )

			Mesh_Void_Ext = Mesh_1.GroupOnGeom(VoidSurfaces[m], VoidSurfaces_Names[m],SMESH.FACE)

	isDone = Mesh_1.Compute()
	globals().update(locals()) ### This adds all variables created in this function

	return Mesh_1




class TestDimensions(): # This testing input class is valid only for no void case
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

def HandleRC(returncode, Sims={}, AffectedSims=[], MeshName='',MeshErrors=[]):
	'''
	Depending on the returncode, you can either remove the AffectedSims from
	the dictionary of simulations or add the Mesh to the list of mesh errors
	If Mesh errors is populated then VirtualLab will stop before running the simulations.
	'''
	for Sim in AffectedSims:
		Sims.pop(Sim)


def GeomError(Parameters):
	''' This function is imported in during the Setup to pick up any errors which will occur for the given geometrical dimension. i.e. impossible dimensions '''
	# Ensure that pipe length is always greater to or equal to block length
	message = None
	return message

def ErrorHandling(Info, ReturnCode):
	print(ReturnCode)

if __name__ == '__main__':
	if len(sys.argv) == 1:
		Create(TestDimensions())
	# 1 argument provided which is the parameter file
	elif len(sys.argv) == 2:
		ParameterFile = sys.argv[1]
		sys.path.insert(0, os.path.dirname(ParameterFile))
		Parameters = __import__(os.path.splitext(os.path.basename(ParameterFile))[0])
		Create(Parameters)
