import os
import numpy as np
from salome.geom import geomBuilder
from salome.smesh import smeshBuilder
import SMESH
import salome_version
from SalomeFunc import ObjIndex, AddGroup

if salome_version.getVersions()[0] < 9:
	import salome
	theStudy = salome.myStudy
	geompy = geomBuilder.New(theStudy)
	smesh = smeshBuilder.New(theStudy)
else :
	geompy = geomBuilder.New()
	smesh = smeshBuilder.New()

def CreateDomain(objMesh,Parameter):
	O = geompy.MakeVertex(0, 0, 0)
	OX = geompy.MakeVectorDXDYDZ(1, 0, 0)
	OY = geompy.MakeVectorDXDYDZ(0, 1, 0)
	OZ = geompy.MakeVectorDXDYDZ(0, 0, 1)

	### Get the geometry used for the mesh and move it to the desired location
	TestpieceRef = objMesh.GetShape()



	### Get the coil geometry
	if Parameter.CoilType in ('Test','test'):
		CoilWidth = 0.005
		CoilRad =  0.006
		Vertex_Coil = geompy.MakeVertex(0.0, 0.0, 0.025 + Parameter.CoilGap + CoilWidth/2)

		Torus_1 = geompy.MakeTorus(Vertex_Coil, OZ, CoilRad, CoilWidth/2)
		Coil = geompy.MakeCompound([Torus_1])
		geompy.addToStudy( Coil, 'Coil' )

	if Parameter.CoilType in ('New'):
		CoilWidth = 0.002
		Vertex_1 = geompy.MakeVertex(0.02, 0.01, 0.03)
		Vertex_2 = geompy.MakeVertex(0.02, -0.01, 0.03)
		Vertex_3 = geompy.MakeVertex(0, 0.01, 0.03)
		Vertex_4 = geompy.MakeVertex(0, -0.01, 0.03)
		Vertex_5 = geompy.MakeVertex(-0.01, 0, 0.03)
		Line_1 = geompy.MakeLineTwoPnt(Vertex_3, Vertex_1)
		Line_2 = geompy.MakeLineTwoPnt(Vertex_2, Vertex_4)
		Arc_1 = geompy.MakeArc(Vertex_3, Vertex_5, Vertex_4)
		Wire_1 = geompy.MakeWire([Line_1, Line_2, Arc_1], 1e-07)
		Disk_1 = geompy.MakeDiskPntVecR(Vertex_1, OX, CoilWidth)
		Coil = geompy.MakePipe(Disk_1, Wire_1)
		geompy.addToStudy( Wire_1, 'Wire_1' )
		geompy.addToStudy( Disk_1, 'Disk_1' )
		geompy.addToStudy( Coil, 'Coil' )
	
		CoilIn = geompy.CreateGroup(Coil, geompy.ShapeType['FACE'])
		geompy.UnionIDs(CoilIn, [3])
		geompy.addToStudyInFather(Coil, CoilIn, 'CoilIn')

		CoilOut = geompy.CreateGroup(Coil, geompy.ShapeType['FACE'])
		geompy.UnionIDs(CoilOut, [22])
		geompy.addToStudyInFather(Coil, CoilOut, 'CoilOut')

		CoilSurface = geompy.CreateGroup(Coil, geompy.ShapeType['FACE'])
		geompy.UnionIDs(CoilSurface, [3, 7, 12, 17, 22])
		geompy.addToStudyInFather(Coil, CoilSurface, 'CoilSurface')

	if Parameter.CoilType = 'HIVE':
		CoilRef = geompy.ImportSTEP("/home/rhydian/Documents/Scripts/Simulation/virtuallab/Scripts/HIVE/Coils/HIVE COIL.stp", False, True)
		CoilInIx, CoilOutIx = [24], [173]

		cCoilIn = geompy.MakeCDG(geompy.GetSubShape(CoilRef, CoilInIx))
		cCoilOut = geompy.MakeCDG(geompy.GetSubShape(CoilRef, CoilOutIx))
		CoilVect = geompy.MakeVector(cCoilIn, cCoilOut)
		CrdCoilIn = np.array(geompy.PointCoordinates(cCoilIn))
		CrdCoilOut = np.array(geompy.PointCoordinates(cCoilOut))

		CoilTerminal = [0.090915, 0, 0.02]
		Translation = np.array(CoilTerminal) - (CrdCoilIn + CrdCoilOut)/2
		Coil = geompy.MakeTranslation(CoilRef, Translation[0], Translation[1], Translation[2])

		RotateVector = geompy.MakeTranslation(OZ, CoilTerminal[0], CoilTerminal[1], CoilTerminal[2])
		RotateAngle = geompy.GetAngleRadians(CoilVect, OY)
		Coil = geompy.MakeRotation(Coil, RotateVector, -RotateAngle)
		geompy.addToStudy(Coil,'Coil')

		GrpCoil = AddGroup(Coil, 'Coil', geompy.SubShapeAllIDs(Coil, geompy.ShapeType["SOLID"]))
		GrpCoilSurface = AddGroup(Coil, 'CoilSurface', geompy.SubShapeAllIDs(Coil, geompy.ShapeType["FACE"]))
		GrpCoilIn = AddGroup(Coil, 'CoilIn', CoilInIx)
		GrpCoilOut = AddGroup(Coil, 'CoilOut', CoilOutIx)


	### Creating Chamber
	Setup = geompy.MakeCompound([SampleGeom, Coil])
	geompy.addToStudy( Setup, 'Setup' )

	[Xmin,Xmax,Ymin,Ymax,Zmin,Zmax] = geompy.BoundingBox(Setup)
	Dimensions = [Xmax-Xmin,Ymax-Ymin,Zmax-Zmin]
	VacRad = max(Dimensions)*2 ### Width of Vacuum
	Vertex_Air = geompy.MakeVertex(0.5*(Xmin + Xmax),0.5*(Ymin + Ymax),0.5*(Zmin +Zmax))
	Sphere_1 = geompy.MakeSpherePntR(Vertex_Air, VacRad)
	geompy.addToStudy( Sphere_1, 'Sphere' )

	Domain = geompy.MakePartition([Sphere_1], [Setup], [], [], geompy.ShapeType["SOLID"], 0, [], 0)
	geompy.addToStudy( Domain, 'Domain' )
	
	### Add Groups to 'Domain'
	## Vacuum
	# Vacuum Volume
	Vacuumgrp = geompy.CreateGroup(Domain, geompy.ShapeType['SOLID'])
	geompy.UnionIDs(Vacuumgrp, [2])
	geompy.addToStudyInFather( Domain, Vacuumgrp, 'Vacuum' )
	# Vacuum External Surface
	VacuumSurface = geompy.CreateGroup(Domain, geompy.ShapeType['FACE'])
	geompy.UnionIDs(VacuumSurface, [4])
	geompy.addToStudyInFather( Domain, VacuumSurface, 'VacuumSurface' )

	## Coil
	# Coil Volume
	CoilIx = geompy.SubShapeAllIDs(Coil, geompy.ShapeType["SOLID"])
	NewIx = ObjIndex(Domain, Coil, CoilIx)[0]
	Coilgrp = geompy.CreateGroup(Domain, geompy.ShapeType['SOLID'])
	geompy.UnionIDs(Coilgrp, NewIx)
	geompy.addToStudyInFather(Domain, Coilgrp, 'Coil')
	# Coil In
	NewIx = ObjIndex(Domain, Coil, CoilIn.GetSubShapeIndices())[0]
	CoilIn = geompy.CreateGroup(Domain, geompy.ShapeType['FACE'])
	geompy.UnionIDs(CoilIn, NewIx)
	geompy.addToStudyInFather(Domain, CoilIn, 'CoilIn')
	# CoilOut
	NewIx = ObjIndex(Domain, Coil, CoilOut.GetSubShapeIndices())[0]
	CoilOut = geompy.CreateGroup(Domain, geompy.ShapeType['FACE'])
	geompy.UnionIDs(CoilOut, NewIx)
	geompy.addToStudyInFather(Domain, CoilOut, 'CoilOut')
	# Coil Surface
	NewIx = ObjIndex(Domain, Coil, CoilSurface.GetSubShapeIndices())[0]
	CoilSurface = geompy.CreateGroup(Domain, geompy.ShapeType['FACE'])
	geompy.UnionIDs(CoilSurface, NewIx)
	geompy.addToStudyInFather(Domain, CoilSurface, 'CoilSurface')


	### Meshing part ###
	### Start by meshing the whole domain coarsly, and then add sub meshes for the sample and coil
	### This is in essence the sub mesh for the vacuum 

	### Main Mesh
	# Mesh Parameters
	VacArc = np.pi*VacRad
	VacLength = VacArc/20	

	EMMesh1 = smesh.Mesh(Domain)

	Vacuum_1D = EMMesh1.Segment()
	Vacuum_1D_Parameters = Vacuum_1D.LocalLength(VacLength,None,1e-07)

	Vacuum_2D = EMMesh1.Triangle(algo=smeshBuilder.NETGEN_2D)
	Vacuum_2D_Parameters = Vacuum_2D.Parameters()
	Vacuum_2D_Parameters.SetOptimize( 1 )
	Vacuum_2D_Parameters.SetFineness( 3 )
	Vacuum_2D_Parameters.SetChordalError( 0.1 )
	Vacuum_2D_Parameters.SetChordalErrorEnabled( 0 )
	Vacuum_2D_Parameters.SetUseSurfaceCurvature( 1 )
	Vacuum_2D_Parameters.SetQuadAllowed( 0 )
	Vacuum_2D_Parameters.SetMaxSize( VacLength )
	Vacuum_2D_Parameters.SetMinSize( 0.001 )

	Vacuum_3D = EMMesh1.Tetrahedron()
	Vacuum_3D_Parameters = Vacuum_3D.Parameters()
	Vacuum_3D_Parameters.SetOptimize( 1 )
	Vacuum_3D_Parameters.SetFineness( 3 )
	Vacuum_3D_Parameters.SetMaxSize( VacLength )
	Vacuum_3D_Parameters.SetMinSize( 0.001 )

	smesh.SetName(Vacuum_1D_Parameters, 'Vacuum_1D_Parameters')
	smesh.SetName(Vacuum_2D_Parameters, 'Vacuum_2D_Parameters')
	smesh.SetName(Vacuum_3D_Parameters, 'Vacuum_3D_Parameters')

	### Sub Meshes

	## Coil
	# Mesh parameters
	CoilCirc = np.pi*CoilWidth
	CoilLength = CoilCirc/10
	Coil_1D = EMMesh1.Segment(geom=Coilgrp)
	Coil_1D_Parameters = Coil_1D.LocalLength(CoilLength,None,1e-07)

	Coil_2D = EMMesh1.Triangle(algo=smeshBuilder.NETGEN_2D,geom=Coilgrp)
	Coil_2D_Parameters = Coil_2D.Parameters()
	Coil_2D_Parameters.SetOptimize( 1 )
	Coil_2D_Parameters.SetFineness( 3 )
	Coil_2D_Parameters.SetChordalError( 0.1 )
	Coil_2D_Parameters.SetChordalErrorEnabled( 0 )
	Coil_2D_Parameters.SetUseSurfaceCurvature( 1 )
	Coil_2D_Parameters.SetQuadAllowed( 0 )
	Coil_2D_Parameters.SetMaxSize(CoilLength*5 )
	Coil_2D_Parameters.SetMinSize(CoilLength )

	Coil_3D = EMMesh1.Tetrahedron(geom=Coilgrp)
	Coil_3D_Parameters = Coil_3D.Parameters()
	Coil_3D_Parameters.SetOptimize( 1 )
	Coil_3D_Parameters.SetFineness( 3 )
	Coil_3D_Parameters.SetMaxSize( CoilLength*5 )
	Coil_3D_Parameters.SetMinSize( CoilLength )

	Sub_mesh_Coil = Coil_1D.GetSubMesh()
	smesh.SetName(Sub_mesh_Coil, 'Sub-mesh_Coil')
	smesh.SetName(Coil_1D_Parameters, 'Coil_1D_Parameters')
	smesh.SetName(Coil_2D_Parameters, 'Coil_2D_Parameters')
	smesh.SetName(Coil_3D_Parameters, 'Coil_3D_Parameters')


	## Testpiece

	# Solid parts associated with the original testpiece
	SampleIx = geompy.SubShapeAllIDs(SampleGeom, geompy.ShapeType["SOLID"])
	DomainVol = ObjIndex(Domain, SampleGeom, SampleIx)[0]

	if objMesh.GetMeshOrder():
		SubMeshes = objMesh.GetMeshOrder()
	else:
		SubMeshes = objMesh.GetMesh().GetSubMeshes()

	SMlist = []
	for sm in SubMeshes:
		shape = sm.GetSubShape()
		smname = 'Sub_{}'.format(str(shape.GetName()))

		# Create geometrical object
		OldIx = shape.GetSubShapeIndices()
		NewIx, shapetype = ObjIndex(Domain, SampleGeom, OldIx)
		group = geompy.CreateGroup(Domain, geompy.ShapeType[shapetype])
		geompy.UnionIDs(group, NewIx)
		geompy.addToStudyInFather(Domain, group, smname)

		Hypoth = objMesh.GetHypothesisList(shape)
		if ((Hypoth[0]).__class__.__name__) =='_objref_StdMeshers_Regular_1D':
			SM1D = EMMesh1.Segment(geom=group)
		status = EMMesh1.AddHypothesis(Hypoth[1],geom=group)

		if ((Hypoth[2]).__class__.__name__) =='_objref_NETGENPlugin_NETGEN_2D_ONLY':
			SM2D = EMMesh1.Triangle(algo=smeshBuilder.NETGEN_2D,geom=group)
		status = EMMesh1.AddHypothesis(Hypoth[3],geom=group)

		if ((Hypoth[4]).__class__.__name__) =='_objref_NETGENPlugin_NETGEN_3D':
			SM3D = EMMesh1.Tetrahedron(geom=group)
		status = EMMesh1.AddHypothesis(Hypoth[5],geom=group)

		subm = SM1D.GetSubMesh()
		SMlist.append(subm)
		smesh.SetName(subm, smname)

		if shapetype == 'SOLID':
			DomainVol = [x for x in DomainVol if x not in NewIx]
			

	Sub_Sample = geompy.CreateGroup(Domain, geompy.ShapeType['SOLID'])
	geompy.UnionIDs(Sub_Sample, DomainVol)
	geompy.addToStudyInFather(Domain, Sub_Sample, 'Sub_Sample')

	MeshHypoth = objMesh.GetHypothesisList(SampleGeom)
	if ((MeshHypoth[0]).__class__.__name__) =='_objref_StdMeshers_Regular_1D':
		Sample1D = EMMesh1.Segment(geom=Sub_Sample)
	status = EMMesh1.AddHypothesis(MeshHypoth[1],geom=Sub_Sample)

	if ((MeshHypoth[2]).__class__.__name__) == '_objref_NETGENPlugin_NETGEN_2D_ONLY':
		Sample2D = EMMesh1.Triangle(algo=smeshBuilder.NETGEN_2D,geom=Sub_Sample)
	status = EMMesh1.AddHypothesis(MeshHypoth[3],geom=Sub_Sample)

	if ((MeshHypoth[4]).__class__.__name__) == '_objref_NETGENPlugin_NETGEN_3D':
		Sample3D = EMMesh1.Tetrahedron(geom=Sub_Sample)
	status = EMMesh1.AddHypothesis(MeshHypoth[5],geom=Sub_Sample)
	
	MSub_Sample = Sample1D.GetSubMesh()
	smesh.SetName(MSub_Sample, 'Sub_Sample')
	SMlist.append(MSub_Sample)
	SMlist.insert(0,Sub_mesh_Coil)
	
	smesh.SetName(EMMesh1.GetMesh(), 'EMMesh1')
	isDone = EMMesh1.SetMeshOrder([SMlist])

	### Add mesh groups

	## Testpiece groups - these have to be added first

	# Split the groups up in to elements and nodal groups
	MeshGroups = {'Nodal' : [], 'Element' : []}
	for i,grp in enumerate(objMesh.GetGroups()):
		if str(grp.GetType()) == 'NODE':
			MeshGroups['Nodal'].append(grp)
		else: 
			MeshGroups['Element'].append(grp)

	grplist, namelist = [], []
	EMgrplist, EMnamelist = [], []
	for grp in MeshGroups['Element']:
		grpname = str(grp.GetName())
		shape = grp.GetShape()

		OldIx = shape.GetSubShapeIndices()
		NewIx, shapetype = ObjIndex(Domain, SampleGeom, OldIx)

		### Create a group on the geometry
		group = geompy.CreateGroup(Domain, geompy.ShapeType[shapetype])
		geompy.UnionIDs(group, NewIx)
		geompy.addToStudyInFather(Domain, group, grpname)

		### Create a mesh group using the geometry group
		if shapetype == 'SOLID':
			grp = EMMesh1.GroupOnGeom(group,grpname,SMESH.VOLUME)
		elif shapetype == 'FACE':
			grp = EMMesh1.GroupOnGeom(group,grpname,SMESH.FACE)
		elif shapetype == 'EDGE':
			grp = EMMesh1.GroupOnGeom(group,grpname,SMESH.EDGE)


		NodeName = None
		for other in MeshGroups['Nodal']:
			if shape.IsSame(other.GetShape()):
				NodeName = str(other.GetName())
				break

		grplist.append(grp)
		namelist += [NodeName,grpname]

		EMgrplist.append(grp)
		EMnamelist += [None,grpname]

	# Vacuum groups
	EMgrplist.append(EMMesh1.GroupOnGeom(Vacuumgrp,'Vacuum',SMESH.VOLUME))
	EMnamelist += [None,'Vacuum']
	EMgrplist.append(EMMesh1.GroupOnGeom(VacuumSurface,'VacuumSurface',SMESH.FACE))
	EMnamelist += [None,'VacuumSurface']

	# Coil groups
	EMgrplist.append(EMMesh1.GroupOnGeom(Coilgrp,'Coil',SMESH.VOLUME))
	EMnamelist += [None,'Coil']
	EMgrplist.append(EMMesh1.GroupOnGeom(CoilIn,'CoilIn',SMESH.FACE))
	EMnamelist += [None,'CoilIn']
	EMgrplist.append(EMMesh1.GroupOnGeom(CoilOut,'CoilOut',SMESH.FACE))
	EMnamelist += [None,'CoilOut']
	EMgrplist.append(EMMesh1.GroupOnGeom(CoilSurface,'CoilSurface',SMESH.FACE))
	EMnamelist += [None,'CoilSurface']

	isdone = EMMesh1.Compute()

	# Create compound meshes of the parts to ensure the node numberings are the same	
	SampleMesh = smesh.Concatenate(grplist, 1, 1, 1e-05,True,'Sample')
	EMMesh = smesh.Concatenate(EMgrplist, 1, 1, 1e-05,True,'EMMesh')

	# Remove unwanted node groups
	for i,grp in enumerate(SampleMesh.GetGroups()):
		if namelist[i]:
			grp.SetName(namelist[i])
		else :
			SampleMesh.RemoveGroup(grp)

	for i,grp in enumerate(EMMesh.GetGroups()):
		if EMnamelist[i]:
			grp.SetName(EMnamelist[i])
		else :
			EMMesh.RemoveGroup(grp)


	print('\n############# Ignore these messages ###################')
	isDone = SampleMesh.Compute()
	isDone = EMMesh.Compute()
	print('###############################################\n')

	return SampleMesh, EMMesh



