import sys
sys.dont_write_bytecode=True
import numpy as np
import os
from types import SimpleNamespace
from Scripts.Common.VLFunctions import VerifyParameters

'''
This script generates a 'RVE'  using the SALOME software package.
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
	import os
	
	import  SMESH
	import salome_version
	from Scripts.VLPackages.Salome import SalomeFunc

	geompy = geomBuilder.New()
	smesh = smeshBuilder.New()


	part=[]
	Box_2 = geompy.MakeBoxDXDYDZ(8000/4, 8000/4, 8000/4)

	Translation_3 = geompy.MakeTranslation(Box_2, -2000/2, -2000/2, -150)
	geompy.addToStudy( Translation_3, 'Translation_3' )

	part.append(Translation_3)
	
	
	cornea = {}
	sphere={}
	cornea1 = {}
	sphere1={}
	f = open((getattr(Parameters,'rve',None)),"r")
	

	n = 1

	for l in f:
    
    	    x, y, z,r = [ float(v) for v in l.split() ]
    	    pt = geompy.MakeVertex(x, y, z)
    	    cornea[n] = pt
    	    geompy.addToStudy(pt, "cornea_%s"%(n))
    	    sp=geompy.MakeSpherePntR(pt, r)
    	    sphere[n] = sp
    	    geompy.addToStudy(sp, "sphere_%s"%(n))
    	    part.append(sp)
    	    n += 1
    	    pass

	f = open((getattr(Parameters,'rveos',None)),"r")
	

	n1 = 1

	for l in f:
    
    	    x, y, z,r = [ float(v) for v in l.split() ]
    	    pt = geompy.MakeVertex(x, y, z)
    	    cornea1[n1] = pt
    	    geompy.addToStudy(pt, "cornea1_%s"%(n1))
    	    sp=geompy.MakeSpherePntR(pt, r)
    	    sphere1[n1] = sp
    	    geompy.addToStudy(sp, "sphere1_%s"%(n1))
    	    part.append(sp)
    	    n1 += 1
    	    pass
	Partition_1 = geompy.MakePartition(part, [], [], [], geompy.ShapeType["SOLID"], 0, [], 0)
	geompy.addToStudy( Partition_1, 'Partition_1' )

	r=geompy.ExtractShapes(Partition_1, geompy.ShapeType["SOLID"], False)
	eid = geompy.GetSubShapeID(Partition_1, r[0])

	vol= geompy.GetSubShape(Partition_1, [eid])
	geompy.addToStudyInFather(Partition_1, vol, "vol")

	mo=[]
	r1=geompy.ExtractShapes(Partition_1, geompy.ShapeType["FACE"], False)
	spherep1=[]
	for i in range(0,6):
    	    eid1=geompy.GetSubShapeID(Partition_1, r1[i])
    	    Face= geompy.GetSubShape(Partition_1, [eid1])
    	    geompy.addToStudyInFather(Partition_1, Face, "face_%s"%(i))
    	    spherep1.append(eid1)
    	    
	fix1s = geompy.CreateGroup(Partition_1, geompy.ShapeType["FACE"])
	geompy.UnionIDs(fix1s, spherep1)
	geompy.addToStudyInFather( Partition_1, fix1s, 'fix1s' )

	spherep=[]
	for i in range(1,n):
    	    eid1=geompy.GetSubShapeID(Partition_1, r[i])
    	    Face= geompy.GetSubShape(Partition_1, [eid1])
    	    geompy.addToStudyInFather(Partition_1, Face, "sphere_%s"%(i))
    	    spherep.append(eid1)
    	    mo.append(eid1)
	rhe = geompy.CreateGroup(Partition_1, geompy.ShapeType["SOLID"])
	geompy.UnionIDs(rhe, spherep)
	geompy.addToStudyInFather( Partition_1, rhe, 'rhe' )


	spherep2=[]
	for i in range(1,n1):
    	    eid2=geompy.GetSubShapeID(Partition_1, r[i])
    	    Face= geompy.GetSubShape(Partition_1, [eid2])
    	    geompy.addToStudyInFather(Partition_1, Face, "sphere1_%s"%(i))
    	    spherep2.append(eid2)
    	    mo.append(eid2)
	os = geompy.CreateGroup(Partition_1, geompy.ShapeType["SOLID"])
	geompy.UnionIDs(os, spherep2)
	geompy.addToStudyInFather( Partition_1, os, 'os' )

	
	eid = geompy.GetSubShapeID(Partition_1, r[0])
	mo.append(eid)
	vol1 = geompy.CreateGroup(Partition_1, geompy.ShapeType["SOLID"])
	geompy.UnionIDs(vol1, mo)
	geompy.addToStudyInFather( Partition_1, vol1, 'vol1' )
	


	smesh = smeshBuilder.New()
#smesh.SetEnablePublish( False ) # Set to False to avoid publish in study if not needed or in some particular situations:
                                 # multiples meshes built in parallel, complex and numerous mesh edition (performance)

	Mesh_1 = smesh.Mesh(Partition_1)
	NETGEN_2D3D_1 = smesh.CreateHypothesis('NETGEN_2D3D', 'libNETGENEngine.so')
	NETGEN_3D_Parameters_1 = smesh.CreateHypothesis('NETGEN_Parameters', 'NETGENEngine')
	
	NETGEN_3D_Parameters_1.SetMaxSize(300 )
	NETGEN_3D_Parameters_1.SetMinSize( 5 )
	NETGEN_3D_Parameters_1.SetSecondOrder( 0 )
	NETGEN_3D_Parameters_1.SetOptimize( 1 )
	NETGEN_3D_Parameters_1.SetFineness( 5 )
	NETGEN_3D_Parameters_1.SetGrowthRate( 0.1 )
	NETGEN_3D_Parameters_1.SetNbSegPerEdge( 15 )
	NETGEN_3D_Parameters_1.SetNbSegPerRadius( 2 )
	NETGEN_3D_Parameters_1.SetChordalError( -1 )
	NETGEN_3D_Parameters_1.SetChordalErrorEnabled( 0 )
	NETGEN_3D_Parameters_1.SetUseSurfaceCurvature( 1 )
	NETGEN_3D_Parameters_1.SetFuseEdges( 1 )
	NETGEN_3D_Parameters_1.SetQuadAllowed( 0 )
	NETGEN_3D_Parameters_1.SetCheckChartBoundary( 240 )

	status = Mesh_1.AddHypothesis(NETGEN_3D_Parameters_1)
	status = Mesh_1.AddHypothesis(NETGEN_2D3D_1)
	fix1s = Mesh_1.GroupOnGeom(fix1s,'fix1s',SMESH.FACE)
	vol = Mesh_1.GroupOnGeom(vol,'vol',SMESH.VOLUME)
	rhe = Mesh_1.GroupOnGeom(rhe,'rhe',SMESH.VOLUME)
	os = Mesh_1.GroupOnGeom(os,'os',SMESH.VOLUME)
	vol1 = Mesh_1.GroupOnGeom(vol1,'vol1',SMESH.VOLUME)
	isDone = Mesh_1.Compute()
	Mesh_1.Scale( Mesh_1, SMESH.PointStruct ( 0, 0, 0 ), [ 2.789, 2.789, 2.789 ], 0 )
	Mesh_1.Scale( Mesh_1, SMESH.PointStruct ( 0, 0, 0 ), [ 0.0001, 0.0001, 0.0001 ], 0 )
	Mesh_1.Scale( Mesh_1, SMESH.PointStruct ( 0, 0, 0 ), [ 0.001, 0.001, 0.001 ], 0 )
	return Mesh_1

if __name__ == '__main__':
	if len(sys.argv) == 1:
		Create(Example())
	# 1 argument provided which is the parameter file
	elif len(sys.argv) == 2:
		ParameterFile = sys.argv[1]
		sys.path.insert(0, os.path.dirname(ParameterFile))
		Parameters = __import__(os.path.splitext(os.path.basename(ParameterFile))[0])
		Create(VL,Parameters)
