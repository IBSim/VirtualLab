import sys
import os
from salome.geom import geomBuilder
from salome.smesh import smeshBuilder
import  SMESH
import salome_version
from Scripts.Common.VLPackages.Salome import SalomeFunc
import numpy as np

if salome_version.getVersions()[0] < 9:
    import salome
    theStudy = salome.myStudy
    geompy = geomBuilder.New(theStudy)
    smesh = smeshBuilder.New(theStudy)
else :
    geompy = geomBuilder.New()
    smesh = smeshBuilder.New()

O = geompy.MakeVertex(0, 0, 0)
OX = geompy.MakeVectorDXDYDZ(1, 0, 0)
OY = geompy.MakeVectorDXDYDZ(0, 1, 0)
OZ = geompy.MakeVectorDXDYDZ(0, 0, 1)

geompy.addToStudy( O, 'O' )
geompy.addToStudy( OX, 'OX' )
geompy.addToStudy( OY, 'OY' )
geompy.addToStudy( OZ, 'OZ' )

def Test():
    ### Geometry ###

    O = geompy.MakeVertex(0, 0, 0)
    OX = geompy.MakeVectorDXDYDZ(1, 0, 0)
    OY = geompy.MakeVectorDXDYDZ(0, 1, 0)
    OZ = geompy.MakeVectorDXDYDZ(0, 0, 1)
    geompy.addToStudy( O, 'O' )
    geompy.addToStudy( OX, 'OX' )
    geompy.addToStudy( OY, 'OY' )
    geompy.addToStudy( OZ, 'OZ' )

    CoilWidth = 0.005
    Vertex_1 = geompy.MakeVertex(0.1, 0.01, 0.00)
    Vertex_2 = geompy.MakeVertex(0.1, -0.01, 0.00)
    Vertex_3 = geompy.MakeVertex(0, 0.01, 0.00)
    Vertex_4 = geompy.MakeVertex(0, -0.01, 0.00)
    Vertex_5 = geompy.MakeVertex(-0.01, 0, 0.00)
    Line_1 = geompy.MakeLineTwoPnt(Vertex_3, Vertex_1)
    Line_2 = geompy.MakeLineTwoPnt(Vertex_2, Vertex_4)
    Arc_1 = geompy.MakeArc(Vertex_3, Vertex_5, Vertex_4)
    Wire_1 = geompy.MakeWire([Line_1, Line_2, Arc_1], 1e-07)
    geompy.addToStudy(Wire_1,'Wire')
    Disk_1 = geompy.MakeDiskPntVecR(Vertex_1, OX, CoilWidth/2)
    Coil = geompy.MakePipe(Disk_1, Wire_1)
    geompy.addToStudy( Coil, 'Coil' )

    GrpCoil = SalomeFunc.AddGroup(Coil, 'Coil', [1])
    GrpCoilIn = SalomeFunc.AddGroup(Coil, 'CoilIn', [3])
    GrpCoilOut = SalomeFunc.AddGroup(Coil, 'CoilOut', [22])
    GrpCoilSurface = SalomeFunc.AddGroup(Coil, 'CoilSurface', geompy.SubShapeAllIDs(Coil, geompy.ShapeType["FACE"]))

    ### Mesh ###
    # Number of segments which define a circle
    CircSeg = 20
    CoilWidth = 0.005
    Mesh1D = np.pi*CoilWidth/CircSeg
    Mesh2D = Mesh1D
    Mesh3D = Mesh1D

    CoilMesh = smesh.Mesh(Coil)
    Coil_1D = CoilMesh.Segment()
    Coil_1D_Parameters = Coil_1D.LocalLength(Mesh1D, None, 1e-07)

    Coil_2D = CoilMesh.Triangle(algo=smeshBuilder.NETGEN_2D)
    Coil_2D_Parameters = Coil_2D.Parameters()
    Coil_2D_Parameters.SetOptimize( 1 )
    Coil_2D_Parameters.SetFineness( 3 )
    Coil_2D_Parameters.SetChordalError( 0.1 )
    Coil_2D_Parameters.SetChordalErrorEnabled( 0 )
    Coil_2D_Parameters.SetUseSurfaceCurvature( 1 )
    Coil_2D_Parameters.SetQuadAllowed( 0 )
    Coil_2D_Parameters.SetMaxSize(Mesh2D)
    Coil_2D_Parameters.SetMinSize(Mesh2D)

    Coil_3D = CoilMesh.Tetrahedron()
    Coil_3D_Parameters = Coil_3D.Parameters()
    Coil_3D_Parameters.SetOptimize( 1 )
    Coil_3D_Parameters.SetFineness( 3 )
    Coil_3D_Parameters.SetMaxSize(Mesh3D)
    Coil_3D_Parameters.SetMinSize(Mesh3D)

    smesh.SetName(CoilMesh.GetMesh(), 'Coil_Mesh')
    smesh.SetName(Coil_1D_Parameters, 'Coil_1D_Parameters')
    smesh.SetName(Coil_2D_Parameters, 'Coil_2D_Parameters')
    smesh.SetName(Coil_3D_Parameters, 'Coil_3D_Parameters')

    CoilMesh.GroupOnGeom(GrpCoil,'Coil',SMESH.VOLUME)
    CoilMesh.GroupOnGeom(GrpCoilIn,'CoilIn',SMESH.FACE)
    CoilMesh.GroupOnGeom(GrpCoilOut,'CoilOut',SMESH.FACE)
    CoilMesh.GroupOnGeom(GrpCoilSurface,'CoilSurface',SMESH.FACE)

    # =========================================================================
    # Make refrence vector
    Centre = geompy.MakeVertex(0.01, 0, 0)
    Orientation = {'Centre':Centre,'System':[OX,OY,OZ]}

    return CoilMesh, Orientation

def Test_Hollow():
    ### Geometry ###

    O = geompy.MakeVertex(0, 0, 0)
    OX = geompy.MakeVectorDXDYDZ(1, 0, 0)
    OY = geompy.MakeVectorDXDYDZ(0, 1, 0)
    OZ = geompy.MakeVectorDXDYDZ(0, 0, 1)
    geompy.addToStudy( O, 'O' )
    geompy.addToStudy( OX, 'OX' )
    geompy.addToStudy( OY, 'OY' )
    geompy.addToStudy( OZ, 'OZ' )

    CoilWidth = 0.005
    CoilThickness = 0.0005
    Vertex_1 = geompy.MakeVertex(0.1, 0.01, 0.03)
    Vertex_2 = geompy.MakeVertex(0.1, -0.01, 0.03)
    Vertex_3 = geompy.MakeVertex(0, 0.01, 0.03)
    Vertex_4 = geompy.MakeVertex(0, -0.01, 0.03)
    Vertex_5 = geompy.MakeVertex(-0.01, 0, 0.03)
    Line_1 = geompy.MakeLineTwoPnt(Vertex_3, Vertex_1)
    Line_2 = geompy.MakeLineTwoPnt(Vertex_2, Vertex_4)
    Arc_1 = geompy.MakeArc(Vertex_3, Vertex_5, Vertex_4)
    Wire_1 = geompy.MakeWire([Line_1, Line_2, Arc_1], 1e-07)
    geompy.addToStudy(Wire_1,'Wire')
    Disk_1 = geompy.MakeDiskPntVecR(Vertex_1, OX, CoilWidth/2)
    Disk_2 = geompy.MakeDiskPntVecR(Vertex_1, OX, CoilWidth/2 - CoilThickness)
    Disk_3 = geompy.MakeCutList(Disk_1,[Disk_2],True)
    Coil = geompy.MakePipe(Disk_3, Wire_1)
    geompy.addToStudy( Coil, 'Coil' )

    GrpCoil = SalomeFunc.AddGroup(Coil, 'Coil', [1])
    GrpCoilIn = SalomeFunc.AddGroup(Coil, 'CoilIn', [3])
    GrpCoilOut = SalomeFunc.AddGroup(Coil, 'CoilOut', [40])
    GrpCoilSurface = SalomeFunc.AddGroup(Coil, 'CoilSurface', geompy.SubShapeAllIDs(Coil, geompy.ShapeType["FACE"]))

    ### Mesh ###
    # Number of segments which define a circle
    CircSeg = 20
    CoilWidth = 0.005
    Mesh1D = np.pi*CoilWidth/CircSeg
    Mesh2D = Mesh1D
    Mesh3D = Mesh1D

    CoilMesh = smesh.Mesh(Coil)
    Coil_1D = CoilMesh.Segment()
    Coil_1D_Parameters = Coil_1D.LocalLength(Mesh1D, None, 1e-07)

    Coil_2D = CoilMesh.Triangle(algo=smeshBuilder.NETGEN_2D)
    Coil_2D_Parameters = Coil_2D.Parameters()
    Coil_2D_Parameters.SetOptimize( 1 )
    Coil_2D_Parameters.SetFineness( 3 )
    Coil_2D_Parameters.SetChordalError( 0.1 )
    Coil_2D_Parameters.SetChordalErrorEnabled( 0 )
    Coil_2D_Parameters.SetUseSurfaceCurvature( 1 )
    Coil_2D_Parameters.SetQuadAllowed( 0 )
    Coil_2D_Parameters.SetMaxSize(Mesh2D)
    Coil_2D_Parameters.SetMinSize(Mesh2D)

    Coil_3D = CoilMesh.Tetrahedron()
    Coil_3D_Parameters = Coil_3D.Parameters()
    Coil_3D_Parameters.SetOptimize( 1 )
    Coil_3D_Parameters.SetFineness( 3 )
    Coil_3D_Parameters.SetMaxSize(Mesh3D)
    Coil_3D_Parameters.SetMinSize(Mesh3D)

    smesh.SetName(CoilMesh.GetMesh(), 'Coil_Mesh')
    smesh.SetName(Coil_1D_Parameters, 'Coil_1D_Parameters')
    smesh.SetName(Coil_2D_Parameters, 'Coil_2D_Parameters')
    smesh.SetName(Coil_3D_Parameters, 'Coil_3D_Parameters')

    CoilMesh.GroupOnGeom(GrpCoil,'Coil',SMESH.VOLUME)
    CoilMesh.GroupOnGeom(GrpCoilIn,'CoilIn',SMESH.FACE)
    CoilMesh.GroupOnGeom(GrpCoilOut,'CoilOut',SMESH.FACE)
    CoilMesh.GroupOnGeom(GrpCoilSurface,'CoilSurface',SMESH.FACE)

    # =========================================================================
    # Make refrence vector
    Centre = geompy.MakeVertex(0.01, 0, 0)
    Orientation = {'Centre':Centre,'System':[OX,OY,OZ]}

    return CoilMesh, Orientation

def HIVE(dirname=None):
    if not dirname:
        dirname = os.path.dirname(os.path.abspath(__file__))
    Coil = geompy.ImportSTEP("{}/CoilGeom/HIVE_coil.stp".format(dirname), False, True)
    Coil = geompy.MakeGlueFaces(Coil, 1e-07)
    geompy.addToStudy(Coil,'Coil')

    GrpCoil = SalomeFunc.AddGroup(Coil, 'Coil', geompy.SubShapeAllIDs(Coil, geompy.ShapeType["SOLID"]))
    GrpCoilIn = SalomeFunc.AddGroup(Coil, 'CoilIn', [20])
    GrpCoilOut = SalomeFunc.AddGroup(Coil, 'CoilOut', [40])
    AllFaces = geompy.SubShapeAllIDs(Coil, geompy.ShapeType["FACE"])
    IntFaces = [4,24,44,60,64,80]
    ExtFaces = list(set(AllFaces) - set(IntFaces))
    GrpCoilSurface = SalomeFunc.AddGroup(Coil, 'CoilSurface', ExtFaces)

    ### Mesh ###
    # Number of segments which define a circle
    CircSeg = 16
    CoilWidth = 0.005
    Mesh1D = np.pi*CoilWidth/CircSeg
    Mesh2D = Mesh1D
    Mesh3D = Mesh1D

    CoilMesh = smesh.Mesh(Coil)
    Coil_1D = CoilMesh.Segment()
    Coil_1D_Parameters = Coil_1D.LocalLength(Mesh1D, None, 1e-07)

    Coil_2D = CoilMesh.Triangle(algo=smeshBuilder.NETGEN_2D)
    Coil_2D_Parameters = Coil_2D.Parameters()
    Coil_2D_Parameters.SetOptimize( 1 )
    Coil_2D_Parameters.SetFineness( 3 )
    Coil_2D_Parameters.SetChordalError( 0.1 )
    Coil_2D_Parameters.SetChordalErrorEnabled( 0 )
    Coil_2D_Parameters.SetUseSurfaceCurvature( 1 )
    Coil_2D_Parameters.SetQuadAllowed( 0 )
    Coil_2D_Parameters.SetMaxSize(Mesh2D)
    Coil_2D_Parameters.SetMinSize(Mesh2D)

    Coil_3D = CoilMesh.Tetrahedron()
    Coil_3D_Parameters = Coil_3D.Parameters()
    Coil_3D_Parameters.SetOptimize( 1 )
    Coil_3D_Parameters.SetFineness( 3 )
    Coil_3D_Parameters.SetMaxSize(Mesh3D)
    Coil_3D_Parameters.SetMinSize(Mesh3D)

    smesh.SetName(CoilMesh.GetMesh(), 'Coil_Mesh')
    smesh.SetName(Coil_1D_Parameters, 'Coil_1D_Parameters')
    smesh.SetName(Coil_2D_Parameters, 'Coil_2D_Parameters')
    smesh.SetName(Coil_3D_Parameters, 'Coil_3D_Parameters')

    CoilMesh.GroupOnGeom(GrpCoil,'Coil',SMESH.VOLUME)
    CoilMesh.GroupOnGeom(GrpCoilIn,'CoilIn',SMESH.FACE)
    CoilMesh.GroupOnGeom(GrpCoilOut,'CoilOut',SMESH.FACE)
    CoilMesh.GroupOnGeom(GrpCoilSurface,'CoilSurface',SMESH.FACE)

    # =========================================================================
    # Make refrence vector
    Centre = geompy.MakeVertex(0, 0.005, 0)
    OX_reverse = geompy.MakeVectorDXDYDZ(-1, 0, 0)
    Orientation = {'Centre':Centre,'System':[OY,OX_reverse,OZ]}

    return CoilMesh, Orientation

def HIVE_old(dirname=None):
    if not dirname:
        dirname = os.path.dirname(os.path.abspath(__file__))
    Coil = geompy.ImportSTEP("{}/CoilGeom/HIVE_coil_old.stp".format(dirname), False, True)
    geompy.addToStudy(Coil,'Coil')

    GrpCoil = SalomeFunc.AddGroup(Coil, 'Coil', geompy.SubShapeAllIDs(Coil, geompy.ShapeType["SOLID"]))
    GrpCoilIn = SalomeFunc.AddGroup(Coil, 'CoilIn', [24])
    GrpCoilOut = SalomeFunc.AddGroup(Coil, 'CoilOut', [173])
    GrpCoilSurface = SalomeFunc.AddGroup(Coil, 'CoilSurface', geompy.SubShapeAllIDs(Coil, geompy.ShapeType["FACE"]))

    ### Mesh ###
    # Number of segments which define a circle
    CircSeg = 16
    CoilWidth = 0.005
    Mesh1D = np.pi*CoilWidth/CircSeg
    Mesh2D = Mesh1D
    Mesh3D = Mesh1D

    CoilMesh = smesh.Mesh(Coil)
    Coil_1D = CoilMesh.Segment()
    Coil_1D_Parameters = Coil_1D.LocalLength(Mesh1D, None, 1e-07)

    Coil_2D = CoilMesh.Triangle(algo=smeshBuilder.NETGEN_2D)
    Coil_2D_Parameters = Coil_2D.Parameters()
    Coil_2D_Parameters.SetOptimize( 1 )
    Coil_2D_Parameters.SetFineness( 3 )
    Coil_2D_Parameters.SetChordalError( 0.1 )
    Coil_2D_Parameters.SetChordalErrorEnabled( 0 )
    Coil_2D_Parameters.SetUseSurfaceCurvature( 1 )
    Coil_2D_Parameters.SetQuadAllowed( 0 )
    Coil_2D_Parameters.SetMaxSize(Mesh2D)
    Coil_2D_Parameters.SetMinSize(Mesh2D)

    Coil_3D = CoilMesh.Tetrahedron()
    Coil_3D_Parameters = Coil_3D.Parameters()
    Coil_3D_Parameters.SetOptimize( 1 )
    Coil_3D_Parameters.SetFineness( 3 )
    Coil_3D_Parameters.SetMaxSize(Mesh3D)
    Coil_3D_Parameters.SetMinSize(Mesh3D)

    smesh.SetName(CoilMesh.GetMesh(), 'Coil_Mesh')
    smesh.SetName(Coil_1D_Parameters, 'Coil_1D_Parameters')
    smesh.SetName(Coil_2D_Parameters, 'Coil_2D_Parameters')
    smesh.SetName(Coil_3D_Parameters, 'Coil_3D_Parameters')

    CoilMesh.GroupOnGeom(GrpCoil,'Coil',SMESH.VOLUME)
    CoilMesh.GroupOnGeom(GrpCoilIn,'CoilIn',SMESH.FACE)
    CoilMesh.GroupOnGeom(GrpCoilOut,'CoilOut',SMESH.FACE)
    CoilMesh.GroupOnGeom(GrpCoilSurface,'CoilSurface',SMESH.FACE)

    # =========================================================================
    # Make refrence vector
    OX_reverse = geompy.MakeVectorDXDYDZ(-1, 0, 0)
    Orientation = {'Centre':O,'System':[OY,OX_reverse,OZ]}

    return CoilMesh, Orientation

def Pancake(dirname=None):
    if not dirname:
        dirname = os.path.dirname(os.path.abspath(__file__))
    Coil = geompy.ImportSTEP("{}/CoilGeom/Pancake_coil.step".format(dirname), False, True)
    Coil = geompy.MakeGlueFaces(Coil, 1e-07)
    geompy.addToStudy(Coil,'Coil')

    GrpCoil = SalomeFunc.AddGroup(Coil, 'Coil', geompy.SubShapeAllIDs(Coil, geompy.ShapeType["SOLID"]))
    GrpCoilIn = SalomeFunc.AddGroup(Coil, 'CoilIn', [13])
    GrpCoilOut = SalomeFunc.AddGroup(Coil, 'CoilOut', [31])
    AllFaces = geompy.SubShapeAllIDs(Coil, geompy.ShapeType["FACE"])
    GrpCoilSurface = SalomeFunc.AddGroup(Coil, 'CoilSurface', AllFaces)

    ### Mesh ###
    # Number of segments which define a circle
    CircSeg = 16
    CoilWidth = 0.00586
    Mesh1D = np.pi*CoilWidth/CircSeg
    Mesh2D = Mesh1D
    Mesh3D = Mesh1D

    CoilMesh = smesh.Mesh(Coil)
    Coil_1D = CoilMesh.Segment()
    Coil_1D_Parameters = Coil_1D.LocalLength(Mesh1D, None, 1e-07)

    Coil_2D = CoilMesh.Triangle(algo=smeshBuilder.NETGEN_2D)
    Coil_2D_Parameters = Coil_2D.Parameters()
    Coil_2D_Parameters.SetOptimize( 1 )
    Coil_2D_Parameters.SetFineness( 3 )
    Coil_2D_Parameters.SetChordalError( 0.1 )
    Coil_2D_Parameters.SetChordalErrorEnabled( 0 )
    Coil_2D_Parameters.SetUseSurfaceCurvature( 1 )
    Coil_2D_Parameters.SetQuadAllowed( 0 )
    Coil_2D_Parameters.SetMaxSize(Mesh2D)
    Coil_2D_Parameters.SetMinSize(Mesh2D)

    Coil_3D = CoilMesh.Tetrahedron()
    Coil_3D_Parameters = Coil_3D.Parameters()
    Coil_3D_Parameters.SetOptimize( 1 )
    Coil_3D_Parameters.SetFineness( 3 )
    Coil_3D_Parameters.SetMaxSize(Mesh3D)
    Coil_3D_Parameters.SetMinSize(Mesh3D)

    smesh.SetName(CoilMesh.GetMesh(), 'Coil_Mesh')
    smesh.SetName(Coil_1D_Parameters, 'Coil_1D_Parameters')
    smesh.SetName(Coil_2D_Parameters, 'Coil_2D_Parameters')
    smesh.SetName(Coil_3D_Parameters, 'Coil_3D_Parameters')

    CoilMesh.GroupOnGeom(GrpCoil,'Coil',SMESH.VOLUME)
    CoilMesh.GroupOnGeom(GrpCoilIn,'CoilIn',SMESH.FACE)
    CoilMesh.GroupOnGeom(GrpCoilOut,'CoilOut',SMESH.FACE)
    CoilMesh.GroupOnGeom(GrpCoilSurface,'CoilSurface',SMESH.FACE)

    OX_reverse = geompy.MakeVectorDXDYDZ(-1, 0, 0)
    Orientation = {'Centre':O,'System':[OY,OX_reverse,OZ]}

    return CoilMesh, Orientation

def Coils(Name):
    if Name.lower()=='test':
        return Test()
    elif Name.lower()=='test_hollow':
        return Test_Hollow()
    elif Name.lower()=='hive':
        return HIVE()
    elif Name.lower()=='hive_old':
        return HIVE_old()
    elif Name.lower()=='pancake':
        return Pancake()


if __name__ == "__main__":
    Test()
    # Test_Hollow()
    HIVE(dirname=os.path.dirname(sys.argv[0]))
