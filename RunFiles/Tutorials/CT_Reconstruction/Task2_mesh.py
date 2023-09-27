#!/usr/bin/env python3
#===============================================================================
# Header
#===============================================================================

import sys
sys.dont_write_bytecode=True
from types import SimpleNamespace as Namespace
from Scripts.Common.VirtualLab import VLSetup

def get_mesh():
    #===============================================================================
    # Parameters
    #===============================================================================

    Mesh = Namespace()
    # New mesh with longer "turned" pipe
    Mesh.Name = "AMAZE_turn"
    Mesh.File = 'Monoblock_Turn'
    # This file must be in Scripts/$SIMULATION/PreProc
    # Geometrical Dimensions for Fundamental Hive Sample
    Mesh.BlockWidth = 0.045 #x
    Mesh.BlockLength = 0.045 #y
    Mesh.BlockHeight = 0.035 #z
    Mesh.PipeDiam = 0.0127 #Pipe inner diameter
    Mesh.PipeThick = 0.00415 #Pipe wall thickness
    Mesh.PipeLength = 0.20
    Mesh.TileCentre = [0,0]
    Mesh.TileWidth = Mesh.BlockWidth
    Mesh.TileLength = 0.045 #y
    Mesh.TileHeight = 0.010 #z
    Mesh.PipeCentre = [0, Mesh.TileHeight/2]
    Mesh.TurnDiam = 0.01588 #Pipe outer diameter in turned section
    Mesh.TurnLength = 0.03 #Length of turned section

    # Mesh parameters
    Mesh.Length1D = 0.003
    Mesh.Length2D = 0.003
    Mesh.Length3D = 0.003

    Mesh.SubTile = [0.001, 0.001, 0.001]
    Mesh.CoilFace = 0.0006
    Mesh.Fillet = 0.0005
    Mesh.Deflection = 0.01
    Mesh.PipeSegmentN = 24 # Number of segments for pipe circumference

    Parameters_Master = Namespace(Mesh=Mesh)

    #===============================================================================
    # Environment
    #===============================================================================

    VirtualLab=VLSetup('HIVE','Tutorials')
    VirtualLab.Parameters(Parameters_Master)
    VirtualLab.Mesh()

