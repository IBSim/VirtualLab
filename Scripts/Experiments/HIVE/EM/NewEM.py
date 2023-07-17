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
from EM.CoilDesigns import Coils

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

class CoilFOR(object):
    ''' Class for frame of reference for coil system'''
    def __init__(self, coil, centre, system):
        self.coil = coil
        self.centre = centre
        self.system = [geompy.MakeLine(centre,dir) for dir in system]
        self.centre_coord()
        self.system_coord()

    def translation(self,DX,DY,DZ):
        # Move coil geom
        self.coil = geompy.MakeTranslation(self.coil,DX,DY,DZ )
        # Move FOR
        self.centre = geompy.MakeTranslation(self.centre,DX,DY,DZ)
        for i in range(len(self.system)):
            self.system[i] = geompy.MakeTranslation(self.system[i],DX,DY,DZ)

        # update vale of centre (system not affected by translation)
        self.centre_coord()


    def rotation(self,axis,angle):
        # rotate coil
        self.coil = geompy.MakeRotation(self.coil,axis,angle)
        # Rotate FOR
        self.centre = geompy.MakeRotation(self.centre,axis,angle)
        for i in range(len(self.system)):
            self.system[i] = geompy.MakeRotation(self.system[i],axis,angle)

        # updtae value of centre and system
        self.centre_coord()
        self.system_coord()

    def centre_coord(self):
        # get coordinate for centre point
        self.centre_val = np.array(geompy.PointCoordinates(self.centre))

    def system_coord(self):
        # get direction of system
        self.system_val = np.array([GetDirection(s) for s in self.system])

def GetDirection(norm):
    V1,V2 = geompy.SubShapeAll(norm,geompy.ShapeType['VERTEX'])
    Crd_V1 = np.array(geompy.PointCoordinates(V1))
    Crd_V2 = np.array(geompy.PointCoordinates(V2))
    return Crd_V2 - Crd_V1

def GetAngle(dir1,dir2):
    product = np.dot(dir1,dir2)
    mag1,mag2 = np.linalg.norm(dir1),np.linalg.norm(dir2)
    theta_rad = np.arccos(product/(mag1*mag2))
    return theta_rad


def GEOM_Create(SampleSurface,SampleCentre,CoilVect,PipeVect,
                CoilGeom, CoilCentre, CoilSystem, Parameters):



    O = geompy.MakeVertex(0, 0, 0)
    OX = geompy.MakeVectorDXDYDZ(1, 0, 0)
    OY = geompy.MakeVectorDXDYDZ(0, 1, 0)
    OZ = geompy.MakeVectorDXDYDZ(0, 0, 1)
    geompy.addToStudy( O, 'O' )
    geompy.addToStudy( OX, 'OX' )
    geompy.addToStudy( OY, 'OY' )
    geompy.addToStudy( OZ, 'OZ' )

    print('\nCreating ERMES geometry\n')

    # ==========================================================================
    # get orientation information and make a class containing this and the coil geometry
    if CoilCentre is None: CoilCentre=O # assume coil centre is at origin
    if CoilSystem is None: CoilSystem=[OX,OY,OZ] # assume terminal is in OX direction,
    CoilRef = CoilFOR(CoilGeom,CoilCentre,CoilSystem)

    # ==========================================================================
    # Move coil so that frames of reference line up
    print('Lining up coil and component frame of reference')
    # Move coil so that it's centre point is as SampleCentre
    gSampleCentre = geompy.MakeVertex(*SampleCentre)
    geompy.addToStudy(gSampleCentre,'SampleCentre')
    CoilRef.translation(*(SampleCentre - CoilRef.centre_val))
    geompy.addToStudy( CoilRef.coil, 'Coil' )

    # Check angle between the coil normal and coil refrence
    # third component of coil system must line up with coil vector
    coil_angle = GetAngle(CoilRef.system_val[2],CoilVect)
    if np.mod(coil_angle,np.pi):
        # get direction normal to both vectors
        norm = np.cross(CoilRef.system_val[2],CoilVect)
        RotateVector = geompy.MakeVector(gSampleCentre,geompy.MakeVertex(*(SampleCentre + norm)))
        CoilRef.rotation(RotateVector,coil_angle)
        geompy.addToStudy(RotateVector,'RotateVector_2')

    # Check angle between the pipe and coil refrence
    # second component of coil system must line up with the pipe
    pipe_angle = GetAngle(CoilRef.system_val[1],PipeVect)
    if np.mod(pipe_angle,np.pi):
        norm = np.cross(CoilRef.system_val[1],PipeVect) # get direction normal to both vectors
        RotateVector = geompy.MakeVector(gSampleCentre,geompy.MakeVertex(*(SampleCentre + norm)))
        CoilRef.rotation(RotateVector,pipe_angle)
        geompy.addToStudy(RotateVector,'RotateVector_1')

    # Get coil tight to top of sample
    SampleBB = geompy.BoundingBox(SampleSurface)
    CoilBB = geompy.BoundingBox(geompy.MakeBoundingBox(CoilRef.coil,True))
    translation = np.array([0,0,SampleBB[5] - CoilBB[4]])
    CoilRef.translation(*translation)

    # ==========================================================================
    # Move and rotate coil to match experimental setup
    print('Position coil using CoilDisplacement and CoilRotation')
    CoilRef.translation(*Parameters.CoilDisplacement)

    if hasattr(Parameters,'CoilRotation'):
        for r,d in zip(Parameters.CoilRotation,CoilRef.system):
            if r==0: continue
            CoilRef.rotation(d,np.deg2rad(r))

    # ==========================================================================
    # Coil manipulation complete.
    Coil = CoilRef.coil # get final coil geometry from coilFOR
    geompy.addToStudy( Coil, 'Coil' )
    geompy.addToStudy(CoilRef.centre,'CoilCentre')

    # Check for intersection of coil and sample
    # Make samplesurface shell into solid object
    gm = geompy.MakeShell(SampleSurface)
    Sample_solid = geompy.MakeSolid([gm])
    Common = geompy.MakeCommonList([Sample_solid,Coil])
    Measure = np.array(geompy.BasicProperties(Common))
    Common.Destroy()
    if not all(Measure < 1e-9):
        # If there is an overlap these measurements will be non-zero
        return 2319

    # ==========================================================================
    # Make chamber which includes the sample, coil and vacuum between
    # Make Vacuum around sample
    print('Create Vacuum around coil and sample')
    VacuumSize = getattr(Parameters,'VacuumSize',0.2) # lengthscale associated with the vacuum
    vac_shape = getattr(Parameters,'VacuumShape','sphere')

    if vac_shape.lower() in ('cube','square'):
        Vertex1 = geompy.MakeVertex(*(SampleCentre-VacuumSize))
        Vertex2 = geompy.MakeVertex(*(SampleCentre+VacuumSize))
        Vacuum_orig = geompy.MakeBoxTwoPnt(Vertex1, Vertex2)
        VacSurfaceIx = [3,13,23,27,31,33]
    elif vac_shape.lower() in ('sphere','circle'):
        Vacuum_orig = geompy.MakeSpherePntR(gSampleCentre, VacuumSize)
        VacSurfaceIx = [3]

    # cut sample geom from vacuum
    Vacuum = geompy.MakeCutList(Vacuum_orig, [Sample_solid], True)
    # partition vacuum with coil geometry
    Chamber = geompy.MakePartition([Vacuum], [Coil], [], [], geompy.ShapeType["SOLID"], 0, [], 0)
    geompy.addToStudy( Chamber, 'Chamber' )
    # ==========================================================================
    # add groups to Chamber
    # Sample surface
    SampleSurfaceIx = geompy.SubShapeAllIDs(SampleSurface,geompy.ShapeType['FACE'])
    Ix = SalomeFunc.ObjIndex(Chamber, SampleSurface, SampleSurfaceIx, Strict=True)[0]
    geomSampleSurface = SalomeFunc.AddGroup(Chamber, 'SampleSurface', Ix)
    # vacuum surface
    Ix = SalomeFunc.ObjIndex(Chamber, Vacuum_orig, VacSurfaceIx)[0]
    geomVacuumSurface = SalomeFunc.AddGroup(Chamber, 'VacuumSurface', Ix)
    # vacuum (solid)
    geomVacuum = SalomeFunc.AddGroup(Chamber, 'Vacuum', [2])
    # coil
    coil_ix = geompy.SubShapeAllIDs(CoilGeom, geompy.ShapeType["SOLID"])
    Ix = SalomeFunc.ObjIndex(Chamber, Coil, coil_ix, Strict=False)[0]
    geomCoil = SalomeFunc.AddGroup(Chamber, 'Coil', Ix)

    # globals().update(locals())
    print('\nERMES geometry created successfully\n')
    return Chamber

def MESH_Create(Chamber, SampleMesh, CoilMeshInfo, Parameters):

    print('Creating ERMES Mesh\n')
    VacuumSize = getattr(Parameters,'VacuumSize',0.2)
    VacuumSegment = getattr(Parameters,'VacuumSegment',25)

    # Create dictionary of groups to easily find
    ChamberGroups = geompy.GetExistingSubObjects(Chamber,True)
    GroupDict = {str(grp.GetName()):grp for grp in ChamberGroups}

    # ==========================================================================
    ### Main Mesh
    # Mesh Parameters
    vac_mesh_lengthscale = 2*np.pi*VacuumSize/VacuumSegment # base mesh size based on sphere, but can also be used for cube
    Vacuum1D = getattr(Parameters,'Vacuum1D',vac_mesh_lengthscale)
    Vacuum2D = getattr(Parameters,'Vacuum2D',vac_mesh_lengthscale)
    Vacuum3D = getattr(Parameters,'Vacuum3D',vac_mesh_lengthscale)

    # This will be a mesh only of the coil and vacuum
    print('Adding mesh parameters for vacuum')
    ERMES = smesh.Mesh(Chamber)
    # 1D
    Vacuum_1D = ERMES.Segment()
    Vacuum_1D_Parameters = Vacuum_1D.LocalLength(Vacuum1D,None,1e-07)
    # 2D
    Vacuum_2D = ERMES.Triangle(algo=smeshBuilder.NETGEN_2D)
    Vacuum_2D_Parameters = Vacuum_2D.Parameters()
    Vacuum_2D_Parameters.SetOptimize( 1 )
    Vacuum_2D_Parameters.SetFineness( 2 )
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
    Vacuum_3D_Parameters.SetFineness( 2 )
    Vacuum_3D_Parameters.SetMaxSize( Vacuum3D )
    Vacuum_3D_Parameters.SetMinSize( 0.001 )

    smesh.SetName(ERMES, 'ERMES')
    smesh.SetName(Vacuum_1D_Parameters, 'Vacuum_1D_Parameters')
    smesh.SetName(Vacuum_2D_Parameters, 'Vacuum_2D_Parameters')
    smesh.SetName(Vacuum_3D_Parameters, 'Vacuum_3D_Parameters')

    # Add 'Vacuum' and 'VacuumSurface' groups to mesh
    ERMES.GroupOnGeom(GroupDict['VacuumSurface'], 'VacuumSurface', SMESH.FACE)
    ERMES.GroupOnGeom(GroupDict['Vacuum'], 'Vacuum', SMESH.VOLUME)

    # ==========================================================================
    # Ensure conformal mesh at sample surface
    print('Adding component surface mesh')
    meshSampleSurface = SampleMesh.GetGroupByName('SampleSurface')
    Import_1D2D = ERMES.UseExisting2DElements(geom=GroupDict['SampleSurface'])
    Source_Faces_1 = Import_1D2D.SourceFaces(meshSampleSurface,0,0)

    SampleSub = Import_1D2D.GetSubMesh()
    smesh.SetName(SampleSub, 'Sample')


    # ==========================================================================
    print('Add coil mesh parameters')
    # Get hypothesis used in original coil mesh
    Param1D = CoilMeshInfo.MainMesh.get('Regular_1D', None)
    Param2D = CoilMeshInfo.MainMesh.get('NETGEN_2D_ONLY', None)
    Param3D = CoilMeshInfo.MainMesh.get('NETGEN_3D', None)

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
    ERMES.AddHypothesis(Param1D, geom=GroupDict['Coil'])
    ERMES.AddHypothesis(Param2D, geom=GroupDict['Coil'])
    ERMES.AddHypothesis(Param3D, geom=GroupDict['Coil'])

    CoilSub = ERMES.GetSubMesh(GroupDict['Coil'],'')
    smesh.SetName(CoilSub, 'Coil')

    # CoilOrder.append(CoilSub)
    # ERMES.SetMeshOrder([[SampleSub]])
    # Add groups from
    for grptype, grpdict in CoilMeshInfo.Groups.items():
        for Name, Ix in grpdict.items():
            NewIx = SalomeFunc.ObjIndex(Chamber, GroupDict['Coil'], Ix,Strict=False)[0]
            grp = SalomeFunc.AddGroup(Chamber, Name, NewIx)
            ERMES.GroupOnGeom(grp, Name, getattr(SMESH, grptype))


    # Compute the mesh for the coil and vacuum
    print('Compute mesh')
    ERMES.Compute()

    # Combine the mesh of the sample with the coil & vacuum. This is the mesh used by ERMES
    ERMESmesh = smesh.Concatenate([SampleMesh.GetMesh(),ERMES.GetMesh()], 1, 1, 1e-05, False, 'ERMES')

    # globals().update(locals()) # Useful for dev work
    print('\nERMES mesh created successfully\n')
    return ERMESmesh

def Pipe_terminal(PipeIn,PipeOut):
    cPipeIn = geompy.MakeCDG(PipeIn)
    cPipeOut = geompy.MakeCDG(PipeOut)
    CrdPipeIn = np.array(geompy.PointCoordinates(cPipeIn))
    CrdPipeOut = np.array(geompy.PointCoordinates(cPipeOut))
    return CrdPipeIn, CrdPipeOut

def main():
    DataDict = SalomeFunc.GetArgs()
    Parameters = DataDict['Parameters']

    # ==========================================================================
    # Get mesh from file
    InputFile = DataDict['InputFile']
    (SampleMesh, status) = smesh.CreateMeshesFromMED(InputFile)
    SampleMesh=SampleMesh[0]

    # ==========================================================================
    # Get reference frame and geometry of sample

    # Potential solution for ibsim but too slow
    # if SampleCentre is not None and CoilVector is not None and PipeVector is not None:
    #     # Only need the external surface of the sample
    #     tmp_file = "{}/surface.stl".format(os.path.dirname(DataDict['OutputFile']))
    #     meshSampleSurface = SampleMesh.GetGroupByName('SampleSurface')[0]
    #     # export and import
    #     SampleMesh.ExportSTL( tmp_file, 1, meshSampleSurface)
    #     SampleSurface = geompy.ImportSTL(tmp_file )
    #     SampleSurface = geompy.UnionFaces(SampleSurface) # helps speed this up
    #     geompy.addToStudy( SampleSurface, 'SampleSurface' )

    # Get the sample geometry from the .xao file saved alongside the .med file
    XAO = geompy.ImportXAO("{}.xao".format(os.path.splitext(InputFile)[0]))
    SampleGroups = XAO[3]
    GroupDict = {str(grp.GetName()):grp for grp in SampleGroups}

    if 'SampleSurface' in GroupDict:
        SampleSurface = GroupDict['SampleSurface']
        geompy.addToStudy( SampleSurface, 'SampleSurface' )
    else:
        print('SampleSurface is not a group in the geometry')


    # Calculate centre point as mid point along pipe, if its not provided
    SampleCentre = getattr(Parameters,'SampleCentre',None)
    if SampleCentre is None:
        if 'PipeIn' not in GroupDict or 'PipeOut' not in GroupDict:
            print('PipeIn and PipeOut must be defined in geometry of component')
        else:
            CrdPipeIn, CrdPipeOut = Pipe_terminal(GroupDict['PipeIn'],GroupDict['PipeOut'])
            SampleCentre = (CrdPipeIn + CrdPipeOut)/2
    else:
        # Use given value of SampleCentre, set values for crdpipein and out for simplicity
        CrdPipeIn,CrdPipeOut = None,None

    # Calculate direction of coil location relative to sample
    CoilVector = getattr(Parameters,'CoilVector',None)
    if CoilVector is None:
        if 'CoilFace' not in GroupDict:
            print('CoilFace not defined in geometry. CoilVector is undefined')
        else:
            _norm = geompy.GetNormal(GroupDict['CoilFace'])
            CoilVector = GetDirection(_norm)

    # Calculate direction along pipe
    PipeVector = getattr(Parameters,'PipeVector',None)
    if PipeVector is None:
        if CrdPipeIn is None and CrdPipeOut is None:
            CrdPipeIn, CrdPipeOut = Pipe_terminal(GroupDict['PipeIn'],GroupDict['PipeOut'])
        PipeVector = CrdPipeOut - CrdPipeIn

    # ==========================================================================
    # import coil design
    CoilData, Orientation = Coils(Parameters.CoilType)
    CoilMeshInfo = GetMesh(CoilData) # get information about mesh and geometry of coil

    # ==========================================================================
    # create geometry for ERMES
    # information for geometric part
    CoilGeom = CoilMeshInfo.Geom # get geometric component of coil
    CoilCentre = Orientation.get('Centre',None)
    CoilSystem = Orientation.get('System',None)

    ERMESgeom = GEOM_Create(SampleSurface, SampleCentre, CoilVector, PipeVector,
                            CoilGeom, CoilCentre, CoilSystem, Parameters)
    if ERMESgeom == 2319:
        print("\nImpossible configuration: Coil intersects sample\n")
        if not salome.sg.hasDesktop():
            sys.exit()

    # ==========================================================================
    # create mesh from geometry
    ERMESmesh = MESH_Create(ERMESgeom,SampleMesh,CoilMeshInfo,Parameters)
    # export mesh
    if type(ERMESmesh) == salome.smesh.smeshBuilder.Mesh:
        SalomeFunc.MeshExport(ERMESmesh, DataDict['OutputFile'])


if __name__ == '__main__':
    main()
    # TODO: Add in easy geometry & mesh for testing
