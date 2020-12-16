#!/usr/bin/env python

import sys
sys.dont_write_bytecode=True
import os
import salome
import numpy as np
salome.salome_init()
import salome_notebook
notebook = salome_notebook.NoteBook()

import GEOM
from salome.geom import geomBuilder
import math
import  SMESH, SALOMEDS
from salome.smesh import smeshBuilder

import SalomeFunc
import Parameters

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

def EMCreate(**kwargs):
    MeshFile = kwargs['MeshFile']
    XAOfile = "{}.xao".format(os.path.splitext(MeshFile)[0])

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

    XAO = geompy.ImportXAO(XAOfile)
    SampleGeom = XAO[1]
    SampleGroups = XAO[3]
    geompy.addToStudy( SampleGeom, 'SampleGeom' )

    GroupDict = {}
    for grp in SampleGroups:
        GroupDict[str(grp.GetName())] = grp

    cPipeIn = geompy.MakeCDG(GroupDict['PipeIn'])
    cPipeOut = geompy.MakeCDG(GroupDict['PipeOut'])

    PipeVect = geompy.MakeVector(cPipeIn, cPipeOut)
    CoilNorm = geompy.GetNormal(GroupDict['CoilFace'])

    CrdPipeIn = np.array(geompy.PointCoordinates(cPipeIn))
    CrdPipeOut = np.array(geompy.PointCoordinates(cPipeOut))
    PipeMid = (CrdPipeIn + CrdPipeOut)/2

    # from EM import CoilDesigns
    from EM import CoilDesigns
    CoilFnc = getattr(CoilDesigns, Parameters.CoilType)
    CoilMesh = GetMesh(CoilFnc())

    cCoilIn = geompy.MakeCDG(geompy.GetSubShape(CoilMesh.Geom, CoilMesh.Groups['FACE']['CoilIn']))
    cCoilOut = geompy.MakeCDG(geompy.GetSubShape(CoilMesh.Geom, CoilMesh.Groups['FACE']['CoilOut']))
    CoilVect = geompy.MakeVector(cCoilIn, cCoilOut)
    CrdCoilIn = np.array(geompy.PointCoordinates(cCoilIn))
    CrdCoilOut = np.array(geompy.PointCoordinates(cCoilOut))
    CrdCoilMid = (CrdCoilIn + CrdCoilOut)/2

    SampleBB = geompy.BoundingBox(SampleGeom)
    CoilBB = geompy.BoundingBox(geompy.MakeBoundingBox(CoilMesh.Geom,True))

    CoilTight = PipeMid + np.array([0.090915, 0, SampleBB[5]-PipeMid[2] + CrdCoilMid[2]-CoilBB[4]])
    CoilTerminal = CoilTight + np.array(Parameters.CoilDisplacement)
    Translation = CoilTerminal - CrdCoilMid

    Coil = geompy.MakeTranslation(CoilMesh.Geom, *Translation)

    RotateVector = geompy.MakeTranslation(OZ, *CoilTerminal)
    RotateAngle = geompy.GetAngleRadians(CoilVect, PipeVect)
    Coil = geompy.MakeRotation(Coil, RotateVector, -RotateAngle)

    Coil = geompy.MakeRotation(Coil, PipeVect, Parameters.Rotation/180*np.pi)
    geompy.addToStudy( Coil, 'Coil' )

    Common = geompy.MakeCommonList([SampleGeom,Coil], True)
    Measure = np.array(geompy.BasicProperties(Common))
    Common.Destroy()
    if not all(Measure < 1e-9):
        return 2319

    #TODO
    # Compound = geompy.MakeCompound([Sample, Coil])
    # CompoundBB = geompy.BoundingBox(Compound)

    VacuumRad = 0.2
    VertexPipeMid = geompy.MakeVertex(*PipeMid)
    Vacuum = geompy.MakeSpherePntR(VertexPipeMid, VacuumRad)
    Vacuum = geompy.MakeCutList(Vacuum, [SampleGeom], True)

    Chamber = geompy.MakePartition([Vacuum], [Coil], [], [], geompy.ShapeType["SOLID"], 0, [], 0)
    geompy.addToStudy( Chamber, 'Chamber' )

    SampleSurfaceIx = GroupDict['SampleSurface'].GetSubShapeIndices()
    Ix = SalomeFunc.ObjIndex(Chamber, SampleGeom, SampleSurfaceIx, Strict=True)[0]
    geomSampleSurface = SalomeFunc.AddGroup(Chamber, 'SampleSurface', Ix)

    Ix = SalomeFunc.ObjIndex(Chamber, Vacuum, [3])[0]
    geomVacuumSurface = SalomeFunc.AddGroup(Chamber, 'VacuumSurface', Ix)

    geomVacuum = SalomeFunc.AddGroup(Chamber, 'Vacuum', [2])

    #### MESH ####

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

    smesh.SetName(ERMES, 'ERMES')
    smesh.SetName(Vacuum_1D_Parameters, 'Vacuum_1D_Parameters')
    smesh.SetName(Vacuum_2D_Parameters, 'Vacuum_2D_Parameters')
    smesh.SetName(Vacuum_3D_Parameters, 'Vacuum_3D_Parameters')

    ERMES.GroupOnGeom(geomVacuumSurface, 'VacuumSurface', SMESH.FACE)
    ERMES.GroupOnGeom(geomVacuum, 'Vacuum', SMESH.VOLUME)
    ### Sample sub-mesh

    (HIVEMesh, status) = smesh.CreateMeshesFromMED(MeshFile)
    HIVEMesh=HIVEMesh[0]
    meshSampleSurface = HIVEMesh.GetGroupByName('SampleSurface')
    Import_1D2D = ERMES.UseExisting2DElements(geom=geomSampleSurface)
    Source_Faces_1 = Import_1D2D.SourceFaces(meshSampleSurface,0,0)

    SampleSub = Import_1D2D.GetSubMesh()
    smesh.SetName(SampleSub, 'Sample')

    ### Coil sub-mesh & related groups

    # Coil Mesh parameters which will be added as a sub-mesh
    # CoilOrder = []
    Ix = SalomeFunc.ObjIndex(Chamber, Coil, CoilMesh.MainMesh['Ix'], Strict=False)[0]
    Geom = geompy.GetSubShape(Chamber, Ix)

    Param1D = CoilMesh.MainMesh.get('Regular_1D', None)
    Coil1D = ERMES.Segment(geom=Geom)
    ERMES.AddHypothesis(Param1D, geom=Geom)

    Param2D = CoilMesh.MainMesh.get('NETGEN_2D_ONLY', None)
    Coil2D = ERMES.Triangle(algo=smeshBuilder.NETGEN_2D,geom=Geom)
    ERMES.AddHypothesis(Param2D, geom=Geom)

    Param3D = CoilMesh.MainMesh.get('NETGEN_3D', None)
    Coil3D = ERMES.Tetrahedron(geom=Geom)
    ERMES.AddHypothesis(Param3D, geom=Geom)

    CoilSub = Coil1D.GetSubMesh()
    smesh.SetName(CoilSub, 'Coil')
    # CoilOrder.append(CoilSub)
    # ERMES.SetMeshOrder([CoilOrder])

    # Groups for Coil
    for grptype, grpdict in CoilMesh.Groups.items():
    	for Name, Ix in grpdict.items():
    		NewIx = SalomeFunc.ObjIndex(Chamber, Coil, Ix,Strict=False)[0]
    		grp = SalomeFunc.AddGroup(Chamber, Name, NewIx)
    		ERMES.GroupOnGeom(grp, Name, getattr(SMESH, grptype))

    ERMES.Compute()

    ERMESmesh = smesh.Concatenate([HIVEMesh.GetMesh(),ERMES.GetMesh()], 1, 1, 1e-05, False, 'ERMES')

    SalomeFunc.MeshExport(ERMESmesh, kwargs['OutFile'])

if __name__ == '__main__':
    kwargs = SalomeFunc.GetArgs(sys.argv[1:])
    err = EMCreate(**kwargs)
    if err == 2319:
        sys.exit("Impossible configuration")

# if salome.sg.hasDesktop():
#   salome.sg.updateObjBrowser()
