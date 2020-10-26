import os
import numpy as np
from salome.geom import geomBuilder
from salome.smesh import smeshBuilder
import SMESH
import salome_version
import salome
import sys
import importlib

if salome_version.getVersions()[0] < 9:
	theStudy = salome.myStudy
	geompy = geomBuilder.New(theStudy)
	smesh = smeshBuilder.New(theStudy)
else :
	geompy = geomBuilder.New()
	smesh = smeshBuilder.New()

### This is a file with includes useful SalomeMeca functions. This directory is added to the pythonpath during installation meaning this module can be accessed inside SM

def MeshInfo(Meshpath,outfile):
	([Mesh], status) = smesh.CreateMeshesFromMED(Meshpath+'.med')
	NbNodes = Mesh.NbNodes()
	NbElems = Mesh.NbElements()
	NbEdges = Mesh.NbEdges()
	NbFaces = Mesh.NbFaces()
	NbVolumes = Mesh.NbVolumes()

	string ='Nodes = {}\n'.format(NbNodes) + \
		'Edges = {}\n'.format(NbEdges) + \
		'Faces = {}\n'.format(NbFaces) + \
		'Volumes = {}\n'.format(NbVolumes) + \
		'Elements = {}\n'.format(NbElems)

	if os.path.isdir(outfile):
		outfile = '{}/{}.py'.format(outfile,os.path.splitext(os.path.basename(Meshpath))[0])

	g = open('{}'.format(outfile),'w+')
	g.write(string)
	g.close()

def MeshExport(Mesh,Meshfile, **kwargs):
	Overwrite = kwargs.get('Overwrite',1)
	err = Mesh.ExportMED( Meshfile, auto_groups=0, minor=40, overwrite=Overwrite,meshPart=None,autoDimension=1)
	if not err:
		print("Nodes: {}\nVolumes: {}\nSurfaces: {}\nEdges: {}".format(Mesh.NbNodes(),Mesh.NbVolumes(),Mesh.NbFaces(),Mesh.NbEdges()))
		print ("Mesh '{}' successfully exported to file {}".format(Mesh.GetName(), Meshfile))
	else:
		print("Error in Exporting mesh")

def ObjIndex(NewGeom, OldGeom, OldIndex, Tol=1e-9, Strict=True):
	### This functions finds the index of a shape in a new geometry which was was created from a previous geometry
	NewIndex = []
	ObjectType = str(geompy.GetSubShape(OldGeom,[OldIndex[0]]).GetShapeType())
	DomainGeoms = geompy.SubShapeAll(NewGeom, geompy.ShapeType[ObjectType])
	NumGeoms = len(DomainGeoms)

	if ObjectType == 'VERTEX':
		for Ix in OldIndex:
			OldCoor = np.array(geompy.PointCoordinates(geompy.GetSubShape(OldGeom,[Ix])))
			for shape in DomainGeoms:
				Dist = np.linalg.norm(np.array(geompy.PointCoordinates(shape)) - OldCoor)
				if Dist < Tol:
					NewIndex += shape.GetSubShapeIndices()
					break
	else:
		# Want to check the higher order spatial measure for a shape, i.e. volume for a solid, area for a face
		if ObjectType == 'SOLID': CheckIx=2
		elif ObjectType == 'FACE': CheckIx=1
		else : CheckIx=0

		for Ix in OldIndex:
			# Get the object from the index and get the higher order spatial measure
			obj = geompy.GetSubShape(OldGeom,[Ix])
			OldMeasure = geompy.BasicProperties(obj)[CheckIx]

			for shape in DomainGeoms:
				NewMeasure = geompy.BasicProperties(shape)[CheckIx]
				Check = abs(NewMeasure - OldMeasure) < Tol
				# Check the measurement of each shape with the original to see if they match. If they do
				# make an intersection of both shapes to check if they are in the same place
				if Check and Strict:
					intersect = geompy.MakeCommonList([obj, shape], True)
					IntMeasure = geompy.BasicProperties(intersect)[CheckIx]
					intersect.UnRegister()

					# If the measure of the intersection matches with the original we are confident
					# it's the same shape and so we return this index
					if abs(IntMeasure - OldMeasure) < Tol:
						NewIndex += shape.GetSubShapeIndices()
						break

				elif Check and not Strict:
					objCOM = np.array(geompy.PointCoordinates(geompy.MakeCDG(obj)))
					shapeCOM = np.array(geompy.PointCoordinates(geompy.MakeCDG(shape)))
					if np.linalg.norm(objCOM-shapeCOM) < Tol:
						NewIndex += shape.GetSubShapeIndices()
						break

		for shape in DomainGeoms:
			shape.UnRegister()

	return NewIndex, ObjectType

def AddGroup(Geom, GroupName, Index):
	from salome.geom import geomBuilder
	geompy = geomBuilder.New()

	GroupType = str(geompy.GetSubShape(Geom,Index[0:1]).GetShapeType())
	Name = geompy.CreateGroup(Geom, geompy.ShapeType[GroupType])
	geompy.UnionIDs(Name, Index)
	geompy.addToStudyInFather( Geom, Name, GroupName )
	return Name

def GetArgs(argv):
	ArgDict = {}
	for arg in argv:
		key, value = arg.split('=')
		ArgDict[key]=value
	return ArgDict

def Reload(name):
	importlib.reload(sys.modules[name])

def MeshStore(MeshRn,MeshFile,RCfile,**kwargs):
    if type(MeshRn)==salome.smesh.smeshBuilder.Mesh:
        isDone = MeshRn.Compute()
        MeshExport(MeshRn,MeshFile)
    elif type(MeshRn) == int:
        MeshRC(RCfile,MeshRn)
        # Write this to file to be picked up

def MeshRC(RCfile,returncode):
	with open(RCfile,'w') as f:
		f.write(str(returncode))
