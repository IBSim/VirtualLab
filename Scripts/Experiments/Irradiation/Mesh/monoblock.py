import sys
sys.dont_write_bytecode=True
import numpy as np
import os
from types import SimpleNamespace
from Scripts.Common.VLFunctions import VerifyParameters

'''
This script generates a 'Tungsten monoblock' sample using the SALOME software package.

'''

def Example():
	
	Parameters = SimpleNamespace(Name='Test')
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


	return error,warning
def Create(Parameters):
	from salome.geom import geomBuilder
	from salome.smesh import smeshBuilder
	import  SMESH
	import salome_version
	from Scripts.VLPackages.Salome import SalomeFunc

	if salome_version.getVersions()[0] < 9:
		import salome
		theStudy = salome.myStudy
		geompy = geomBuilder.New(theStudy)
		smesh = smeshBuilder.New(theStudy)
	else :
		geompy = geomBuilder.New()
		smesh = smeshBuilder.New()


	

	geompy = geomBuilder.New()
	Warmour_thickness1=(getattr(Parameters,"Warmour_thickness",None)) # Thickness of the monoblock
	Warmour_thickness=10*Warmour_thickness1 # convert from cm to mm

	# Define origin
	O = geompy.MakeVertex(0, 0, 0)
	OX = geompy.MakeVectorDXDYDZ(1, 0, 0)
	OY = geompy.MakeVectorDXDYDZ(0, 1, 0)
	OZ = geompy.MakeVectorDXDYDZ(0, 0, 1)

	# Center at distance of half the thickness of the monoblock 
	center1 = geompy.MakeVertex(0, -Warmour_thickness/2, 0)
	# Inner radius of CuCrZr pipe
	pipe_radius1=(getattr(Parameters,"pipe_radius",None))
	pipe_radius=10*pipe_radius1 # convert from cm to mm
	# Tungsten monoblock upper height from the origin
	height1=10*(getattr(Parameters,"Warmour_height_upper",None))
	# Tungsten monoblock lower height from the origin
	height2=10*(getattr(Parameters,"Warmour_height_lower",None))
	# Tungsten monoblock width from the origin
	width=(10/2)*(getattr(Parameters,"Warmour_width",None))
	# CuCrZr pipe thickness
	pipe_thickness=10*(getattr(Parameters,"pipe_thickness",None))
	# Pipe length between monoblocks
	pipe_protrusion=10*(getattr(Parameters,"pipe_protrusion",None))
	# Copper interlayer thickness
	copper_interlayer_thickness=10*(getattr(Parameters,"copper_interlayer_thickness",None))
	# Mesh size of monoblock
	mesh_size=(getattr(Parameters,"mesh_size",None))
	# Mesh size of pipe between monobocks
	prot_mesh=(getattr(Parameters,"prot_mesh",None))
	# Tungsten armour upper height extension # Use this to change the height of the tungsten armour
	arm_ext=10*(getattr(Parameters,"arm_ext",None))
	# Mesh size between tungsten armour corner and copper interlayer 
	seg_diag=(getattr(Parameters,"seg_diag",None))
	geom=geompy.MakeVertex(-width, -Warmour_thickness/2, -height2)
	geom1=geompy.MakeVertex(width, -Warmour_thickness/2, -height2)
	geom2=geompy.MakeVertex(0, Warmour_thickness/2, -height2)
	geom3=geompy.MakeVertex(0, -Warmour_thickness/2, -height2)
	# Define coordinates for inner circle of CuCrZr pipe
	pipe_inner_1 = geompy.MakeVertex(0, -Warmour_thickness/2, pipe_radius)
	pipe_inner_2 = geompy.MakeVertex(pipe_radius, -Warmour_thickness/2, 0)
	pipe_inner_3 = geompy.MakeVertex(0, -Warmour_thickness/2, -pipe_radius)
	pipe_inner_4 = geompy.MakeVertex(-pipe_radius, -Warmour_thickness/2, 0)

	Arc_pipe_inner_1 = geompy.MakeArcCenter(center1,pipe_inner_1, pipe_inner_2,False)
	pipe_center_1 = geompy.MakeVertexOnCurve(Arc_pipe_inner_1, 0.5, True)

	Arc_pipe_inner_2 = geompy.MakeArcCenter(center1,pipe_inner_2, pipe_inner_3,False)
	pipe_center_2 = geompy.MakeVertexOnCurve(Arc_pipe_inner_2, 0.5, True)


	Arc_pipe_inner_3 = geompy.MakeArcCenter(center1,pipe_inner_3, pipe_inner_4,False)
	pipe_center_3 = geompy.MakeVertexOnCurve(Arc_pipe_inner_3, 0.5, True)

	Arc_pipe_inner_4 = geompy.MakeArcCenter(center1,pipe_inner_4, pipe_inner_1,False)
	pipe_center_4 = geompy.MakeVertexOnCurve(Arc_pipe_inner_4, 0.5, True)

	# Define coordinates for outer circle of CuCrZr pipe
	pipe_outer_1 = geompy.MakeVertex(0, -Warmour_thickness/2, pipe_radius+pipe_thickness)
	pipe_outer_2 = geompy.MakeVertex(pipe_radius+pipe_thickness, -Warmour_thickness/2, 0)
	pipe_outer_3 = geompy.MakeVertex(0, -Warmour_thickness/2, -(pipe_radius+pipe_thickness))
	pipe_outer_4 = geompy.MakeVertex(-(pipe_radius+pipe_thickness), -Warmour_thickness/2, 0)

	Arc_pipe_outer_1 = geompy.MakeArcCenter(center1,pipe_outer_1, pipe_outer_2,False)
	pipe_center_outer_1 = geompy.MakeVertexOnCurve(Arc_pipe_outer_1, 0.5, True)

	Arc_pipe_outer_2 = geompy.MakeArcCenter(center1,pipe_outer_2, pipe_outer_3,False)
	pipe_center_outer_2 = geompy.MakeVertexOnCurve(Arc_pipe_outer_2, 0.5, True)


	Arc_pipe_outer_3 = geompy.MakeArcCenter(center1,pipe_outer_3, pipe_outer_4,False)
	pipe_center_outer_3 = geompy.MakeVertexOnCurve(Arc_pipe_outer_3, 0.5, True)

	Arc_pipe_outer_4 = geompy.MakeArcCenter(center1,pipe_outer_4, pipe_outer_1,False)
	pipe_center_outer_4 = geompy.MakeVertexOnCurve(Arc_pipe_outer_4, 0.5, True)


	# Define coordinates for outer circle of copper interlayer
	inte_outer_1 = geompy.MakeVertex(0, -Warmour_thickness/2, pipe_radius+pipe_thickness)
	inte_outer_2 = geompy.MakeVertex(pipe_radius+pipe_thickness, -Warmour_thickness/2, 0)
	inte_outer_3 = geompy.MakeVertex(0, -Warmour_thickness/2, -(pipe_radius+pipe_thickness))
	inte_outer_4 = geompy.MakeVertex(-(pipe_radius+pipe_thickness), -Warmour_thickness/2, 0)

	Arc_inte_outer_1 = geompy.MakeArcCenter(center1,inte_outer_1, inte_outer_2,False)
	inte_center_outer_1 = geompy.MakeVertexOnCurve(Arc_inte_outer_1, 0.5, True)

	Arc_inte_outer_2 = geompy.MakeArcCenter(center1,inte_outer_2, inte_outer_3,False)
	inte_center_outer_2 = geompy.MakeVertexOnCurve(Arc_inte_outer_2, 0.5, True)


	Arc_inte_outer_3 = geompy.MakeArcCenter(center1,inte_outer_3, inte_outer_4,False)
	inte_center_outer_3 = geompy.MakeVertexOnCurve(Arc_inte_outer_3, 0.5, True)

	Arc_inte_outer_4 = geompy.MakeArcCenter(center1,inte_outer_4, inte_outer_1,False)
	inte_center_outer_4 = geompy.MakeVertexOnCurve(Arc_inte_outer_4, 0.5, True)

	# Define coordinates for tungsten armour 
	inte_inner_1 = geompy.MakeVertex(0, -Warmour_thickness/2, pipe_radius+pipe_thickness+copper_interlayer_thickness)
	inte_inner_2 = geompy.MakeVertex(pipe_radius+pipe_thickness+copper_interlayer_thickness, -Warmour_thickness/2, 0)
	inte_inner_3 = geompy.MakeVertex(0, -Warmour_thickness/2, -(pipe_radius+pipe_thickness+copper_interlayer_thickness))
	inte_inner_4 = geompy.MakeVertex(-(pipe_radius+pipe_thickness+copper_interlayer_thickness), -Warmour_thickness/2, 0)

	Arc_inte_inner_1 = geompy.MakeArcCenter(center1,inte_inner_1, inte_inner_2,False)
	inte_center_1 = geompy.MakeVertexOnCurve(Arc_inte_inner_1, 0.5, True)

	Arc_inte_inner_2 = geompy.MakeArcCenter(center1,inte_inner_2, inte_inner_3,False)
	inte_center_2 = geompy.MakeVertexOnCurve(Arc_inte_inner_2, 0.5, True)


	Arc_inte_inner_3 = geompy.MakeArcCenter(center1,inte_inner_3,inte_inner_4,False)
	inte_center_3 = geompy.MakeVertexOnCurve(Arc_inte_inner_3, 0.5, True)

	Arc_inte_inner_4 = geompy.MakeArcCenter(center1,inte_inner_4, inte_inner_1,False)
	inte_center_4 = geompy.MakeVertexOnCurve(Arc_inte_inner_4, 0.5, True)




	pipe_arc_1 = geompy.MakeArcCenter(center1,pipe_center_1, pipe_center_2,False)
	pipe_arc_outer_1 = geompy.MakeArcCenter(center1,pipe_center_outer_1, pipe_center_outer_2,False)


	pipe_arc_2 = geompy.MakeArcCenter(center1,pipe_center_2, pipe_center_3,False)
	pipe_arc_outer_2 = geompy.MakeArcCenter(center1,pipe_center_outer_2, pipe_center_outer_3,False)

	pipe_arc_3 = geompy.MakeArcCenter(center1,pipe_center_3, pipe_center_4,False)
	pipe_arc_outer_3 = geompy.MakeArcCenter(center1,pipe_center_outer_3, pipe_center_outer_4,False)


	pipe_arc_4 = geompy.MakeArcCenter(center1,pipe_center_4, pipe_center_1,False)
	pipe_arc_outer_4 = geompy.MakeArcCenter(center1,pipe_center_outer_4, pipe_center_outer_1,False)




	inte_arc_1 = geompy.MakeArcCenter(center1,inte_center_1, inte_center_2,False)
	inte_arc_outer_1 = geompy.MakeArcCenter(center1,inte_center_outer_1, inte_center_outer_2,False)


	inte_arc_2 = geompy.MakeArcCenter(center1,inte_center_2, inte_center_3,False)
	inte_arc_outer_2 = geompy.MakeArcCenter(center1,inte_center_outer_2, inte_center_outer_3,False)

	inte_arc_3 = geompy.MakeArcCenter(center1,inte_center_3, inte_center_4,False)
	inte_arc_outer_3 = geompy.MakeArcCenter(center1,inte_center_outer_3, inte_center_outer_4,False)


	inte_arc_4 = geompy.MakeArcCenter(center1,inte_center_4, inte_center_1,False)
	inte_arc_outer_4 = geompy.MakeArcCenter(center1,inte_center_outer_4, inte_center_outer_1,False)


	arc1 = geompy.MakeLineTwoPnt(pipe_center_1, pipe_center_outer_1)
	arc2 = geompy.MakeLineTwoPnt(pipe_center_2, pipe_center_outer_2)

	arc3 = geompy.MakeLineTwoPnt(pipe_center_3, pipe_center_outer_3)
	arc4 = geompy.MakeLineTwoPnt(pipe_center_4, pipe_center_outer_4)

	#Creation of wire and faces
	Wire_i1 = geompy.MakeWire([arc1, arc2,pipe_arc_1 , pipe_arc_outer_1], 1e-07)
	Face_i1 = geompy.MakeFaceWires([Wire_i1], 1)

	Wire_i2 = geompy.MakeWire([arc2, arc3,pipe_arc_2 , pipe_arc_outer_2], 1e-07)
	Face_i2 = geompy.MakeFaceWires([Wire_i2], 1)

	Wire_i3 = geompy.MakeWire([arc3, arc4,pipe_arc_3 , pipe_arc_outer_3], 1e-07)
	Face_i3 = geompy.MakeFaceWires([Wire_i3], 1)

	Wire_i4 = geompy.MakeWire([arc4, arc1,pipe_arc_4 , pipe_arc_outer_4], 1e-07)
	Face_i4 = geompy.MakeFaceWires([Wire_i4], 1)
	
	#Extrude the faces to the thickness of monoblock
	Extrusion_f1 = geompy.MakePrismVecH(pipe_arc_1, OY, Warmour_thickness)
	Extrusion_f2 = geompy.MakePrismVecH(pipe_arc_2, OY, Warmour_thickness)
	Extrusion_f3 = geompy.MakePrismVecH(pipe_arc_3, OY, Warmour_thickness)
	Extrusion_f4 = geompy.MakePrismVecH(pipe_arc_4, OY, Warmour_thickness)
	Extrusion_1 = geompy.MakePrismVecH(Face_i1, OY, Warmour_thickness)
	Extrusion_2 = geompy.MakePrismVecH(Face_i2, OY, Warmour_thickness)
	Extrusion_3 = geompy.MakePrismVecH(Face_i3, OY, Warmour_thickness)
	Extrusion_4 = geompy.MakePrismVecH(Face_i4, OY, Warmour_thickness)
	
	
	Shell_1 = geompy.MakeShell([Extrusion_f1, Extrusion_f2, Extrusion_f3, Extrusion_f4])
	Compound_1 = geompy.MakeCompound([Extrusion_1, Extrusion_2, Extrusion_3, Extrusion_4])
	Partition_1 = geompy.MakePartition([Shell_1, Compound_1], [], [], [], geompy.ShapeType["SOLID"], 0, [], 0)
	
	arc11 = geompy.MakeLineTwoPnt(inte_center_1, inte_center_outer_1)
	arc21 = geompy.MakeLineTwoPnt(inte_center_2, inte_center_outer_2)

	arc31 = geompy.MakeLineTwoPnt(inte_center_3, inte_center_outer_3)
	arc41 = geompy.MakeLineTwoPnt(inte_center_4, inte_center_outer_4)

	Wire_i11 = geompy.MakeWire([arc11, arc21,inte_arc_1 , inte_arc_outer_1], 1e-07)
	Face_i11 = geompy.MakeFaceWires([Wire_i11], 1)

	Wire_i21 = geompy.MakeWire([arc21, arc31,inte_arc_2 , inte_arc_outer_2], 1e-07)
	Face_i21 = geompy.MakeFaceWires([Wire_i21], 1)

	Wire_i31 = geompy.MakeWire([arc31, arc41,inte_arc_3 , inte_arc_outer_3], 1e-07)
	Face_i31 = geompy.MakeFaceWires([Wire_i31], 1)

	Wire_i41 = geompy.MakeWire([arc41, arc11,inte_arc_4 , inte_arc_outer_4], 1e-07)
	Face_i41 = geompy.MakeFaceWires([Wire_i41], 1)

	Extrusion_f11 = geompy.MakePrismVecH(inte_arc_1, OY, Warmour_thickness)
	Extrusion_f21 = geompy.MakePrismVecH(inte_arc_2, OY, Warmour_thickness)
	Extrusion_f31 = geompy.MakePrismVecH(inte_arc_3, OY, Warmour_thickness)
	Extrusion_f41 = geompy.MakePrismVecH(inte_arc_4, OY, Warmour_thickness)
	Extrusion_11 = geompy.MakePrismVecH(Face_i11, OY, Warmour_thickness)
	Extrusion_21 = geompy.MakePrismVecH(Face_i21, OY, Warmour_thickness)
	Extrusion_31 = geompy.MakePrismVecH(Face_i31, OY, Warmour_thickness)
	Extrusion_41 = geompy.MakePrismVecH(Face_i41, OY, Warmour_thickness)

	Shell_11 = geompy.MakeShell([Extrusion_f11, Extrusion_f21, Extrusion_f31, Extrusion_f41])
	Compound_11 = geompy.MakeCompound([Extrusion_11, Extrusion_21, Extrusion_31, Extrusion_41])
	Partition_11 = geompy.MakePartition([Shell_11, Compound_11], [], [], [], geompy.ShapeType["SOLID"], 0, [], 0)

	tun_1 = geompy.MakeVertex(width, -Warmour_thickness/2, height1)
	tun_2 = geompy.MakeVertex(-width, -Warmour_thickness/2, height1)
	tun_3 = geompy.MakeVertex(width, -Warmour_thickness/2, -(height2))
	tun_4 = geompy.MakeVertex(-(width), -Warmour_thickness/2, -height2)

	block1 = geompy.MakeLineTwoPnt(inte_center_1, tun_1)
	block2 = geompy.MakeLineTwoPnt(inte_center_2, tun_3)
	block3=geompy.MakeLineTwoPnt(tun_1, tun_3)


	Wire_block1 = geompy.MakeWire([inte_arc_1, block1,block2,block3], 1e-07)
	Face_block1 = geompy.MakeFaceWires([Wire_block1], 1)

	block11 = geompy.MakeLineTwoPnt(inte_center_3, tun_4)

	block31=geompy.MakeLineTwoPnt(tun_3, tun_4)


	block12 = geompy.MakeLineTwoPnt(inte_center_4, tun_2)

	block32=geompy.MakeLineTwoPnt(tun_2, tun_4)



	block33=geompy.MakeLineTwoPnt(tun_2, tun_1)
	# If the lower height of the monoblock is equal to the upper height of monoblock
	g=height2+arm_ext
	if height1==g:
  
   	   Wire_block1 = geompy.MakeWire([inte_arc_1, block1,block2,block3], 1e-07)
   	   Face_block1 = geompy.MakeFaceWires([Wire_block1], 1)

   	   Wire_block2 = geompy.MakeWire([inte_arc_2, block11,block2,block31], 1e-07)
   	   Face_block2 = geompy.MakeFaceWires([Wire_block2], 1)

   	   Wire_block3 = geompy.MakeWire([inte_arc_3, block11,block12,block32], 1e-07)
   	   Face_block3 = geompy.MakeFaceWires([Wire_block3], 1)


   	   Wire_block4 = geompy.MakeWire([inte_arc_4, block33,block12,block1], 1e-07)
   	   Face_block4 = geompy.MakeFaceWires([Wire_block4], 1)

   	   Extrusion_f12 = geompy.MakePrismVecH(Face_block1, OY, Warmour_thickness)
   	   Extrusion_f22 = geompy.MakePrismVecH(Face_block2, OY, Warmour_thickness)
   	   Extrusion_f32 = geompy.MakePrismVecH(Face_block3, OY, Warmour_thickness)
   	   Extrusion_f42 = geompy.MakePrismVecH(Face_block4, OY, Warmour_thickness)
  


   	   Shell_12 = geompy.MakeCompound([Extrusion_f12, Extrusion_f22, Extrusion_f32, Extrusion_f42])


   	   Partition_12 = geompy.MakePartition([Shell_12], [], [], [], geompy.ShapeType["SOLID"], 0, [], 0)

   	   hand_11 = geompy.MakePrismVecH(Face_i1, OY, -pipe_protrusion)
   	   hand_21 = geompy.MakePrismVecH(Face_i2, OY, -pipe_protrusion)
   	   hand_31 = geompy.MakePrismVecH(Face_i3, OY, -pipe_protrusion)
   	   hand_41 = geompy.MakePrismVecH(Face_i4, OY, -pipe_protrusion)

   	   hand_part = geompy.MakeCompound([hand_11, hand_21,hand_31, hand_41])

   	   hand_mirror = geompy.MakeMirrorByPoint(hand_part, O)

	   # Combine CuCrZr pipe, copper interlayer, tungsten armour to create a solid
   	   Partition_final = geompy.MakePartition([geom,geom1,geom2,geom3,Partition_1,hand_part,hand_mirror,Partition_12,Partition_11], [], [], [], geompy.ShapeType["SOLID"], 0, [], 0)
   	   tungsten_arm = geompy.CreateGroup(Partition_final, geompy.ShapeType["SOLID"])
   	   geompy.UnionIDs(tungsten_arm, [320, 234, 296, 268])
   	   copper_inter = geompy.CreateGroup(Partition_final, geompy.ShapeType["SOLID"])
   	   geompy.UnionIDs(copper_inter, [368,358,334,348])
   	   cucrzr_pipe = geompy.CreateGroup(Partition_final, geompy.ShapeType["SOLID"])
   	   geompy.UnionIDs(cucrzr_pipe, [84, 60, 36, 2])
   	   pipe_inner_surface = geompy.CreateGroup(Partition_final, geompy.ShapeType["FACE"])
   	   geompy.UnionIDs(pipe_inner_surface, [28, 90, 52, 76,202,229,219,185,151,161,117,134])


   	   pipe_ext = geompy.CreateGroup(Partition_final, geompy.ShapeType["SOLID"])
   	   geompy.UnionIDs(pipe_ext, [156, 98, 122, 139,190,166,224,207])
   	   top_face_armour = geompy.CreateGroup(Partition_final, geompy.ShapeType["FACE"])
   	   geompy.UnionIDs(top_face_armour, [326])
   	   pipe_end_face= geompy.CreateGroup(Partition_final, geompy.ShapeType["FACE"])
   	   geompy.UnionIDs(pipe_end_face,[205,222,232,188,164,120,137,154])
   	   bottom_face_armour= geompy.CreateGroup(Partition_final, geompy.ShapeType["FACE"])
   	   geompy.UnionIDs(bottom_face_armour,[284])
   	   line = geompy.CreateGroup(Partition_final, geompy.ShapeType["EDGE"])
   	   geompy.UnionIDs(line, [104, 143, 126, 109, 194, 211, 177, 172,148, 131, 114, 102, 199, 182, 170, 216])

   	   p1 = geompy.CreateGroup(Partition_final, geompy.ShapeType["VERTEX"])
   	   geompy.UnionIDs(p1, [250])
   	   p2 = geompy.CreateGroup(Partition_final, geompy.ShapeType["VERTEX"])
   	   geompy.UnionIDs(p2, [281])
   	   p3 = geompy.CreateGroup(Partition_final, geompy.ShapeType["VERTEX"])
   	   geompy.UnionIDs(p3, [290])
   	   p4 = geompy.CreateGroup(Partition_final, geompy.ShapeType["VERTEX"])
   	   geompy.UnionIDs(p4, [280])
   	   line2 = geompy.CreateGroup(Partition_final, geompy.ShapeType["EDGE"])
   	   geompy.UnionIDs(line2, [311, 310,262, 282, 251, 263, 283, 252])
   	   line4 = geompy.CreateGroup(Partition_final, geompy.ShapeType["EDGE"])
   	   geompy.UnionIDs(line4,  [288,286,289,291])
   	   geompy.addToStudy( O, 'O' )
   	   geompy.addToStudy( OX, 'OX' )
   	   geompy.addToStudy( OY, 'OY' )
   	   geompy.addToStudy( OZ, 'OZ' )
   	   geompy.addToStudy( Partition_final, 'Partition_final' )

   	   geompy.addToStudyInFather( Partition_final,tungsten_arm, 'tungsten_arm' )
   	   geompy.addToStudyInFather( Partition_final,copper_inter, 'copper_inter' )
   	   geompy.addToStudyInFather( Partition_final,cucrzr_pipe, 'cucrzr_pipe' )
   	   geompy.addToStudyInFather( Partition_final,pipe_ext, 'pipe_ext' )
   	   geompy.addToStudyInFather( Partition_final,pipe_end_face, 'pipe_end_face' )
   	   geompy.addToStudyInFather( Partition_final,top_face_armour, 'top_face_armour' )
   	   geompy.addToStudyInFather( Partition_final,pipe_inner_surface, 'pipe_inner_surface' )
   	   geompy.addToStudyInFather( Partition_final,bottom_face_armour, 'bottom_face_armour' )
   	   geompy.addToStudyInFather( Partition_final,p1, 'p1' )
   	   geompy.addToStudyInFather( Partition_final,p2, 'p2' )
   	   geompy.addToStudyInFather( Partition_final,p3, 'p3' )
   	
   	   import  SMESH, SALOMEDS
   	   from salome.smesh import smeshBuilder

   	   smesh = smeshBuilder.New()

   	   Mesh_21 = smesh.Mesh(Partition_final)
   	   Regular_1D = Mesh_21.Segment()
   	   Number_of_Segments_1 = Regular_1D.NumberOfSegments(mesh_size)
   	   Quadrangle_2D = Mesh_21.Quadrangle(algo=smeshBuilder.QUADRANGLE)
   	   Hexa_3D = Mesh_21.Hexahedron(algo=smeshBuilder.Hexa)
   	   Regular_1D_1 = Mesh_21.Segment(geom=line)
   	   Regular_1D_4 = Mesh_21.Segment(geom=line4)
   	   Regular_1D_3 = Mesh_21.Segment(geom=line2)
   	   Number_of_Segments_2 = Regular_1D_1.NumberOfSegments(prot_mesh)
   	   m=7.5/seg_diag
  
   	   Number_of_Segments_4 = Regular_1D_4.NumberOfSegments(int(mesh_size/2))
   	   Number_of_Segments_3 = Regular_1D_3.NumberOfSegments(seg_diag)
   	   Sub_mesh_1 = Regular_1D_1.GetSubMesh()
   	   Sub_mesh_4 = Regular_1D_4.GetSubMesh()
   	   Sub_mesh_3 = Regular_1D_3.GetSubMesh()
   	   isDone = Mesh_21.Compute()

   	   tungsten_arm= Mesh_21.GroupOnGeom(tungsten_arm,'tungsten_arm',SMESH.VOLUME)
   	   copper_inter = Mesh_21.GroupOnGeom(copper_inter,'copper_inter',SMESH.VOLUME)
   	   cucrzr_pipe = Mesh_21.GroupOnGeom(cucrzr_pipe,'cucrzr_pipe',SMESH.VOLUME)
   	   pipe_ext = Mesh_21.GroupOnGeom(pipe_ext,'pipe_ext',SMESH.VOLUME)
   	   pipe_end_face = Mesh_21.GroupOnGeom(pipe_end_face,'pipe_end_face',SMESH.FACE)
   	   top_face_armour = Mesh_21.GroupOnGeom(top_face_armour,'top_face_armour',SMESH.FACE)
   	   pipe_inner_surface = Mesh_21.GroupOnGeom(pipe_inner_surface,'pipe_inner_surface',SMESH.FACE)
   	   bottom_face_armour = Mesh_21.GroupOnGeom(bottom_face_armour,'bottom_face_armour',SMESH.FACE)	
   	   p1= Mesh_21.GroupOnGeom(p1,'p1',SMESH.NODE)		
   	   p2= Mesh_21.GroupOnGeom(p2,'p2',SMESH.NODE)
   	   p3= Mesh_21.GroupOnGeom(p3,'p3',SMESH.NODE)	 
   	   p4= Mesh_21.GroupOnGeom(p4,'p4',SMESH.NODE)		
   	    
   	   smesh.SetName(Regular_1D.GetAlgorithm(), 'Regular_1D')
   	   smesh.SetName(Quadrangle_2D.GetAlgorithm(), 'Quadrangle_2D')
   	   smesh.SetName(Hexa_3D.GetAlgorithm(), 'Hexa_3D')
   	   smesh.SetName(pipe_ext, 'pipe_ext')
   	   smesh.SetName(copper_inter, 'copper_inter')
   	   smesh.SetName(pipe_end_face, 'pipe_end_face')
   	   smesh.SetName(cucrzr_pipe, 'cucrzr_pipe')
   	   smesh.SetName(pipe_inner_surface, 'pipe_inner_surface')
   	   smesh.SetName(tungsten_arm, 'tungsten_arm')
   	   smesh.SetName(top_face_armour, 'top_face_armour')
   	   smesh.SetName(bottom_face_armour, 'bottom_face_armour')
   	   smesh.SetName(p1, 'p1')
   	   smesh.SetName(p2, 'p2')
   	   smesh.SetName(p3, 'p3')
   	   smesh.SetName(p4, 'p4')
   	   					 
   	   smesh.SetName(Mesh_21.GetMesh(), 'Mesh_21')
   	   smesh.SetName(Sub_mesh_1, 'Sub-mesh_1')
   	   return Mesh_21
	else:
   	  ext_w=geompy.MakeVertex(-(width), -Warmour_thickness/2, height1+arm_ext)
   	  ext_w1=geompy.MakeVertex((width), -Warmour_thickness/2, height1+arm_ext)
   	  ext_w_line=geompy.MakeLineTwoPnt(tun_2, ext_w)
   	  ext_w1_line=geompy.MakeLineTwoPnt(ext_w1, ext_w)
   	  ext_w2_line=geompy.MakeLineTwoPnt(tun_1, ext_w1)
   	  ext_block1 = geompy.MakeWire([ext_w_line, ext_w1_line,ext_w2_line,block33], 1e-07)
   	  Facext_block1 = geompy.MakeFaceWires([ext_block1], 1)
   	  Wire_block1 = geompy.MakeWire([inte_arc_1, block1,block2,block3], 1e-07)
   	  Face_block1 = geompy.MakeFaceWires([Wire_block1], 1)

   	  Wire_block2 = geompy.MakeWire([inte_arc_2, block11,block2,block31], 1e-07)
   	  Face_block2 = geompy.MakeFaceWires([Wire_block2], 1)

   	  Wire_block3 = geompy.MakeWire([inte_arc_3, block11,block12,block32], 1e-07)
   	  Face_block3 = geompy.MakeFaceWires([Wire_block3], 1)


   	  Wire_block4 = geompy.MakeWire([inte_arc_4, block33,block12,block1], 1e-07)
   	  Face_block4 = geompy.MakeFaceWires([Wire_block4], 1)

   	  Extrusion_f12 = geompy.MakePrismVecH(Face_block1, OY, Warmour_thickness)
   	  Extrusion_f22 = geompy.MakePrismVecH(Face_block2, OY, Warmour_thickness)
   	  Extrusion_f32 = geompy.MakePrismVecH(Face_block3, OY, Warmour_thickness)
   	  Extrusion_f42 = geompy.MakePrismVecH(Face_block4, OY, Warmour_thickness)
   	  Extrusion_f52 = geompy.MakePrismVecH(Facext_block1, OY, Warmour_thickness)


   	  Shell_12 = geompy.MakeCompound([Extrusion_f12, Extrusion_f22, Extrusion_f32, Extrusion_f42,Extrusion_f52])


   	  Partition_12 = geompy.MakePartition([geom,geom1,geom2,geom3,Shell_12], [], [], [], geompy.ShapeType["SOLID"], 0, [], 0)

   	  hand_11 = geompy.MakePrismVecH(Face_i1, OY, -pipe_protrusion)
   	  hand_21 = geompy.MakePrismVecH(Face_i2, OY, -pipe_protrusion)
   	  hand_31 = geompy.MakePrismVecH(Face_i3, OY, -pipe_protrusion)
   	  hand_41 = geompy.MakePrismVecH(Face_i4, OY, -pipe_protrusion)

   	  hand_part = geompy.MakeCompound([hand_11, hand_21,hand_31, hand_41])

   	  hand_mirror = geompy.MakeMirrorByPoint(hand_part, O)


   	  Partition_final = geompy.MakePartition([Partition_1,hand_part,hand_mirror,Partition_12,Partition_11], [], [], [], geompy.ShapeType["SOLID"], 0, [], 0)
   	  tungsten_arm = geompy.CreateGroup(Partition_final, geompy.ShapeType["SOLID"])
   	  geompy.UnionIDs(tungsten_arm, [334, 234, 268, 320,296])
   	  copper_inter = geompy.CreateGroup(Partition_final, geompy.ShapeType["SOLID"])
   	  geompy.UnionIDs(copper_inter, [392, 382, 358, 372])
   	  cucrzr_pipe = geompy.CreateGroup(Partition_final, geompy.ShapeType["SOLID"])
   	  geompy.UnionIDs(cucrzr_pipe, [84, 60, 36, 2])
   	  pipe_inner_surface = geompy.CreateGroup(Partition_final, geompy.ShapeType["FACE"])
   	  geompy.UnionIDs(pipe_inner_surface, [28, 90, 52, 76,202,229,219,185,151,161,117,134])

   	  pipe_ext = geompy.CreateGroup(Partition_final, geompy.ShapeType["SOLID"])
   	  geompy.UnionIDs(pipe_ext, [156, 98, 122, 139,190,166,224,207])
   	  top_face_armour = geompy.CreateGroup(Partition_final, geompy.ShapeType["FACE"])
   	  geompy.UnionIDs(top_face_armour, [343])
   	  pipe_end_face= geompy.CreateGroup(Partition_final, geompy.ShapeType["FACE"])
   	  geompy.UnionIDs(pipe_end_face, [205,222,232,188,164,120,137,154])
   	  bottom_face_armour= geompy.CreateGroup(Partition_final, geompy.ShapeType["FACE"])
   	  geompy.UnionIDs(bottom_face_armour,[284])
   	  line = geompy.CreateGroup(Partition_final, geompy.ShapeType["EDGE"])
   	  geompy.UnionIDs(line, [104, 143, 126, 109, 194, 211, 177, 172,148, 131, 114, 102, 199, 182, 170, 216])
   	  line1 = geompy.CreateGroup(Partition_final, geompy.ShapeType["EDGE"])
   	  geompy.UnionIDs(line1, [342, 341, 352, 353])
   	  line2 = geompy.CreateGroup(Partition_final, geompy.ShapeType["EDGE"])
   	  geompy.UnionIDs(line2, [311, 310, 262, 263, 251, 282, 283, 252])
   	  line4 = geompy.CreateGroup(Partition_final, geompy.ShapeType["EDGE"])
   	  geompy.UnionIDs(line4,  [288,286,289,291])
   	  p1 = geompy.CreateGroup(Partition_final, geompy.ShapeType["VERTEX"])
   	  geompy.UnionIDs(p1, [250])
   	  p2 = geompy.CreateGroup(Partition_final, geompy.ShapeType["VERTEX"])
   	  geompy.UnionIDs(p2, [281])
   	  p3 = geompy.CreateGroup(Partition_final, geompy.ShapeType["VERTEX"])
   	  geompy.UnionIDs(p3, [290])
   	  p4 = geompy.CreateGroup(Partition_final, geompy.ShapeType["VERTEX"])
   	  geompy.UnionIDs(p4, [280])
   	  
   	  geompy.addToStudy( O, 'O' )
   	  geompy.addToStudy( OX, 'OX' )
   	  geompy.addToStudy( OY, 'OY' )
   	  geompy.addToStudy( OZ, 'OZ' )
   	  geompy.addToStudy( Partition_final, 'Partition_final' )

   	  geompy.addToStudyInFather( Partition_final,tungsten_arm, 'tungsten_arm' )
   	  geompy.addToStudyInFather( Partition_final,copper_inter, 'copper_inter' )
   	  geompy.addToStudyInFather( Partition_final,cucrzr_pipe, 'cucrzr_pipe' )
   	  geompy.addToStudyInFather( Partition_final,pipe_ext, 'pipe_ext' )
   	  geompy.addToStudyInFather( Partition_final,pipe_end_face, 'pipe_end_face' )
   	  geompy.addToStudyInFather( Partition_final,top_face_armour, 'top_face_armour' )
   	  geompy.addToStudyInFather( Partition_final,pipe_inner_surface, 'pipe_inner_surface' )
   	  geompy.addToStudyInFather( Partition_final,bottom_face_armour, 'bottom_face_armour' )
   	  geompy.addToStudyInFather( Partition_final,p1, 'p1' )
   	  geompy.addToStudyInFather( Partition_final,p2, 'p2' )
   	  geompy.addToStudyInFather( Partition_final,p3, 'p3' )
   	  geompy.addToStudyInFather( Partition_final,p4, 'p4' )
   	 

   	  import  SMESH, SALOMEDS
   	  from salome.smesh import smeshBuilder

   	  smesh = smeshBuilder.New()

   	  Mesh_21 = smesh.Mesh(Partition_final)
   	  Regular_1D = Mesh_21.Segment()
   	  Number_of_Segments_1 = Regular_1D.NumberOfSegments(mesh_size)
   	  Quadrangle_2D = Mesh_21.Quadrangle(algo=smeshBuilder.QUADRANGLE)
   	  Hexa_3D = Mesh_21.Hexahedron(algo=smeshBuilder.Hexa)
   	  Regular_1D_1 = Mesh_21.Segment(geom=line)
   	  Regular_1D_2 = Mesh_21.Segment(geom=line1)
   	  Regular_1D_3 = Mesh_21.Segment(geom=line2)
   	  Regular_1D_4 = Mesh_21.Segment(geom=line4)
   	  Number_of_Segments_1 = Regular_1D_1.NumberOfSegments(prot_mesh)
   	  m=7.5/seg_diag
   	  Number_of_Segments_2 = Regular_1D_2.NumberOfSegments(int(arm_ext/m)+1)

   	  Number_of_Segments_3 = Regular_1D_3.NumberOfSegments(seg_diag)
   	  Number_of_Segments_4 = Regular_1D_4.NumberOfSegments(int(mesh_size/2))
   	  Sub_mesh_1 = Regular_1D_1.GetSubMesh()
   	  Sub_mesh_2 = Regular_1D_2.GetSubMesh()
   	  Sub_mesh_3 = Regular_1D_3.GetSubMesh()
   	  Sub_mesh_4 = Regular_1D_4.GetSubMesh()
   	  isDone = Mesh_21.Compute()

   	  tungsten_arm = Mesh_21.GroupOnGeom(tungsten_arm,'tungsten_arm',SMESH.VOLUME)
   	  copper_inter = Mesh_21.GroupOnGeom(copper_inter,'copper_inter',SMESH.VOLUME)
   	  cucrzr_pipe = Mesh_21.GroupOnGeom(cucrzr_pipe,'cucrzr_pipe',SMESH.VOLUME)
   	  pipe_ext = Mesh_21.GroupOnGeom(pipe_ext,'pipe_ext',SMESH.VOLUME)
   	  pipe_end_face = Mesh_21.GroupOnGeom(pipe_end_face,'pipe_end_face',SMESH.FACE)
   	  top_face_armour = Mesh_21.GroupOnGeom(top_face_armour,'top_face_armour',SMESH.FACE)
   	  pipe_inner_surface = Mesh_21.GroupOnGeom(pipe_inner_surface,'pipe_inner_surface',SMESH.FACE)
   	  bottom_face_armour = Mesh_21.GroupOnGeom(bottom_face_armour,'bottom_face_armour',SMESH.FACE)
   	  p1= Mesh_21.GroupOnGeom(p1,'p1',SMESH.NODE)		
   	  p2= Mesh_21.GroupOnGeom(p2,'p2',SMESH.NODE)
   	  p3= Mesh_21.GroupOnGeom(p3,'p3',SMESH.NODE)
   	  p4= Mesh_21.GroupOnGeom(p4,'p4',SMESH.NODE)		
   	  
   	  smesh.SetName(Regular_1D.GetAlgorithm(), 'Regular_1D')
   	  smesh.SetName(Quadrangle_2D.GetAlgorithm(), 'Quadrangle_2D')
   	  smesh.SetName(Hexa_3D.GetAlgorithm(), 'Hexa_3D')
   	  smesh.SetName(pipe_ext, 'pipe_ext')
   	  smesh.SetName(copper_inter, 'copper_inter')
   	  smesh.SetName(pipe_end_face, 'pipe_end_face')
   	  smesh.SetName(cucrzr_pipe, 'cucrzr_pipe')
   	  smesh.SetName(pipe_inner_surface, 'pipe_inner_surface')
   	  smesh.SetName(tungsten_arm, 'tungsten_arm')
   	  smesh.SetName(top_face_armour, 'top_face_armour')
   	  smesh.SetName(p1, 'p1')
   	  smesh.SetName(p2, 'p2')
   	  smesh.SetName(p3, 'p3')
   	  smesh.SetName(p4, 'p4')
   	  		
   	  smesh.SetName(bottom_face_armour, 'bottom_face_armour')
   	  smesh.SetName(Mesh_21.GetMesh(), 'Mesh_21')
   	  smesh.SetName(Sub_mesh_1, 'Sub-mesh_1')
   	  return Mesh_21

