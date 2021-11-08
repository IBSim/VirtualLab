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

def EMCreate(SampleMesh, SampleGeom, Parameters):
    # Default parameters
    VacuumRadius = getattr(Parameters,'VacuumRadius',0.2)
    VacuumSegment = getattr(Parameters,'VacuumSegment', 25)


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

    SampleGroups = geompy.GetExistingSubObjects(SampleGeom,True)
    # Create dictionary of groups to easily find
    GroupDict = {str(grp.GetName()):grp for grp in SampleGroups}

    cPipeIn = geompy.MakeCDG(GroupDict['PipeIn'])
    cPipeOut = geompy.MakeCDG(GroupDict['PipeOut'])

    PipeVect = geompy.MakeVector(cPipeIn, cPipeOut)
    CoilNorm = geompy.GetNormal(GroupDict['CoilFace'])

    CrdPipeIn = np.array(geompy.PointCoordinates(cPipeIn))
    CrdPipeOut = np.array(geompy.PointCoordinates(cPipeOut))
    PipeMid = (CrdPipeIn + CrdPipeOut)/2

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

    if True:
        VertexPipeMid = geompy.MakeVertex(*PipeMid)
        Vacuum_orig = geompy.MakeSpherePntR(VertexPipeMid, VacuumRadius)
        gm = geompy.MakeShell(GroupDict["SampleSurface"])
        Solid_1 = geompy.MakeSolid([gm])
        Vacuum = geompy.MakeCutList(Vacuum_orig, [Solid_1], True)
    else:
        pass
        #TODO
        # move centre point of sphere to centre point of bounding box
        # Compound = geompy.MakeCompound([Sample, Coil])
        # CompoundBB = geompy.BoundingBox(Compound)

    Chamber = geompy.MakePartition([Vacuum], [Coil], [], [], geompy.ShapeType["SOLID"], 0, [], 0)
    geompy.addToStudy( Chamber, 'Chamber' )

    SampleSurfaceIx = GroupDict['SampleSurface'].GetSubShapeIndices()
    Ix = SalomeFunc.ObjIndex(Chamber, SampleGeom, SampleSurfaceIx, Strict=True)[0]
    geomSampleSurface = SalomeFunc.AddGroup(Chamber, 'SampleSurface', Ix)

    Ix = SalomeFunc.ObjIndex(Chamber, Vacuum_orig, [3])[0]

    geomVacuumSurface = SalomeFunc.AddGroup(Chamber, 'VacuumSurface', Ix)

    geomVacuum = SalomeFunc.AddGroup(Chamber, 'Vacuum', [2])

    #### MESH ####

    ### Main Mesh
    # Mesh Parameters
    Vacuum1D = getattr(Parameters,'Vacuum1D',2*np.pi*VacuumRadius/VacuumSegment)
    Vacuum2D = getattr(Parameters,'Vacuum2D',Vacuum1D)
    Vacuum3D = getattr(Parameters,'Vacuum3D',Vacuum1D)

    # This will be a mesh only of the coil and vacuum
    ERMES = smesh.Mesh(Chamber)
    # 1D
    Vacuum_1D = ERMES.Segment()
    Vacuum_1D_Parameters = Vacuum_1D.LocalLength(Vacuum1D,None,1e-07)
    # 2D
    Vacuum_2D = ERMES.Triangle(algo=smeshBuilder.NETGEN_2D)
    Vacuum_2D_Parameters = Vacuum_2D.Parameters()
    Vacuum_2D_Parameters.SetOptimize( 1 )
    Vacuum_2D_Parameters.SetFineness( 3 )
    Vacuum_2D_Parameters.SetChordalError( 0.1 )
    Vacuum_2D_Parameters.SetChordalErrorEnabled( 0 )
    Vacuum_2D_Parameters.SetUseSurfaceCurvature( 1 )
    Vacuum_2D_Parameters.SetQuadAllowed( 0 )
    Vacuum_2D_Parameters.SetMaxSize( Vacuum2D )
    Vacuum_2D_Parameters.SetMinSize( 0.001 )
    # 3D
    Vacuum_3D = ERMES.Tetrahedron()
    Vacuum_3D_Parameters = Vacuum_3D.Parameters()
    Vacuum_3D_Parameters.SetOptimize( 1 )
    Vacuum_3D_Parameters.SetFineness( 3 )
    Vacuum_3D_Parameters.SetMaxSize( Vacuum3D )
    Vacuum_3D_Parameters.SetMinSize( 0.001 )

    smesh.SetName(ERMES, 'ERMES')
    smesh.SetName(Vacuum_1D_Parameters, 'Vacuum_1D_Parameters')
    smesh.SetName(Vacuum_2D_Parameters, 'Vacuum_2D_Parameters')
    smesh.SetName(Vacuum_3D_Parameters, 'Vacuum_3D_Parameters')

    # Add 'Vacuum' and 'VacuumSurface' groups to mesh
    ERMES.GroupOnGeom(geomVacuumSurface, 'VacuumSurface', SMESH.FACE)
    ERMES.GroupOnGeom(geomVacuum, 'Vacuum', SMESH.VOLUME)

    # Ensure conformal mesh at sample surface
    meshSampleSurface = SampleMesh.GetGroupByName('SampleSurface')
    Import_1D2D = ERMES.UseExisting2DElements(geom=geomSampleSurface)
    Source_Faces_1 = Import_1D2D.SourceFaces(meshSampleSurface,0,0)

    SampleSub = Import_1D2D.GetSubMesh()
    smesh.SetName(SampleSub, 'Sample')

    ### Coil sub-mesh & related groups
    # Coil Mesh parameters which will be added as a sub-mesh
    Ix = SalomeFunc.ObjIndex(Chamber, Coil, CoilMesh.MainMesh['Ix'], Strict=False)[0]
    Geom = geompy.GetSubShape(Chamber, Ix) # GEOM object of the coil

    # Get hypothesis used in original coil mesh
    Param1D = CoilMesh.MainMesh.get('Regular_1D', None)
    Param2D = CoilMesh.MainMesh.get('NETGEN_2D_ONLY', None)
    Param3D = CoilMesh.MainMesh.get('NETGEN_3D', None)

    # Update hypothesis with values from parameters (if provided)
    if hasattr(Parameters,'Coil1D'):
        Param1D.SetLength(Parameters.Coil1D)

    if hasattr(Parameters,'Coil2D'):
        if type(Parameters.Coil2D) in (int,float):
            Max2D = Min2D = Parameters.Coil2D
        if type(Parameters.Coil2D) in (list,tuple):
            Min2D,Max2D = Parameters.Coil2D[:2]
        Param2D.SetMinSize(Min2D)
        Param2D.SetMaxSize(Max2D)

    if hasattr(Parameters,'Coil3D'):
        if type(Parameters.Coil3D) in (int,float):
            Max3D = Min3D = Parameters.Coil3D
        if type(Parameters.Coil3D) in (list,tuple):
            Min3D,Max3D = Parameters.Coil3D[:2]
        Param3D.SetMinSize(Min3D)
        Param3D.SetMaxSize(Max3D)

    # Apply hypothesis to ERMES mesh
    ERMES.AddHypothesis(Param1D, geom=Geom)
    ERMES.AddHypothesis(Param2D, geom=Geom)
    ERMES.AddHypothesis(Param3D, geom=Geom)

    CoilSub = ERMES.GetSubMesh(Geom,'')
    smesh.SetName(CoilSub, 'Coil')

    # CoilOrder.append(CoilSub)
    # ERMES.SetMeshOrder([[SampleSub]])

    # Add groups from original coil mesh
    for grptype, grpdict in CoilMesh.Groups.items():
        for Name, Ix in grpdict.items():
            NewIx = SalomeFunc.ObjIndex(Chamber, Coil, Ix,Strict=False)[0]
            grp = SalomeFunc.AddGroup(Chamber, Name, NewIx)
            ERMES.GroupOnGeom(grp, Name, getattr(SMESH, grptype))

    # Compute the mesh for the coil and vacuum
    ERMES.Compute()

    # Combine the mesh of the sample with the coil & vacuum. This is the mesh used by ERMES
    ERMESmesh = smesh.Concatenate([SampleMesh.GetMesh(),ERMES.GetMesh()], 1, 1, 1e-05, False, 'ERMES')

    globals().update(locals()) # Useful for dev work

    return ERMESmesh




if __name__ == '__main__':
    #### TODO: Add in easy geometry & mesh for testing

    DataDict = SalomeFunc.GetArgs()
    InputFile = DataDict['InputFile']
    # Get sample mesh from .med file
    (SampleMesh, status) = smesh.CreateMeshesFromMED(InputFile)
    SampleMesh=SampleMesh[0]
    # Get the sample geometry from the .xao file saved alongside the .med file
    XAO = geompy.ImportXAO("{}.xao".format(os.path.splitext(InputFile)[0]))
    SampleGeom, SampleGroups = XAO[1],XAO[3]
    geompy.addToStudy( SampleGeom, 'SampleGeom' )
    for grp in SampleGroups:
        geompy.addToStudyInFather(SampleGeom, grp, str(grp.GetName()))

    # Create ERMES mesh using the sample mesh and geometry
    ERMESmesh = EMCreate(SampleMesh, SampleGeom, DataDict['Parameters'])

    # Export ERMESmesh if mesh type
    if type(ERMESmesh) == salome.smesh.smeshBuilder.Mesh:
        SalomeFunc.MeshExport(ERMESmesh, DataDict['OutputFile'])
    # Check return vaue from EMCreate
    elif ERMESmesh == 2319:
        sys.exit("\nImpossible configuration: Coil intersects sample\n")
