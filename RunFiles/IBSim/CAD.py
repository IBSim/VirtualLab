#!/usr/bin/env python3
import sys
import os
sys.dont_write_bytecode=True
from types import SimpleNamespace as Namespace

from Scripts.Common.VirtualLab import VLSetup

'''
Scripts for running a uniform or ermes analysis using a CAD geometry.
Use the below three flags to dictate the analysis you want to perform
'''

CreateMesh = False
UniformAnalysis = False
ERMESAnalysis = True

# ====================================================================
# Setup VirtualLab & its settings

Simulation = 'HIVE'
Project = 'IBSim'

VirtualLab=VLSetup(
           Simulation,
           Project)

VirtualLab.Settings(Mode='h')

# ====================================================================
# mesh parameters
if CreateMesh:

    Mesh = Namespace()
    Mesh.Name = 'CAD'

    Mesh.File = 'Monoblock_Turn2'
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
    Mesh.ExportGeom = 'stp'

    # Mesh parameters
    Mesh.Length1D = 0.003
    Mesh.Length2D = 0.003
    Mesh.Length3D = 0.003

    Mesh.SubTile = [0.001, 0.001, 0.001]
    Mesh.CoilFace = 0.0006
    Mesh.Fillet = 0.0005
    Mesh.Deflection = 0.01
    Mesh.PipeSegmentN = 24 # Number of segments for pipe circumference

    Main_parameters = Namespace(Mesh = Mesh)
    VirtualLab.Parameters(Main_parameters, None, RunMesh=True)
    
    VirtualLab.Mesh()


# ====================================================================
# mesh parameters
# parameters common to uniform and ermes
Sim = Namespace()
Sim.Mesh = 'CAD'
Sim.Materials = {'Block':'Copper_NL', 'Pipe':'Copper_NL', 'Tile':'Tungsten_NL'}

#############
## Coolant ##
#############
# HTC between coolant and pipe (need Coolant and Pipe properties)
Sim.Pipe = {'Type':'smooth tube', 'Diameter':0.01, 'Length':0.05}
Sim.Coolant = {'Temperature':30, 'Pressure':1, 'Velocity':10}

#############
### Aster ###
#############
Sim.Model = '3D'
Sim.Solver = 'MUMPS'
Sim.TempExt = 20

if UniformAnalysis:
    # copy Sim parameters and add additional values
    Sim_uniform = Namespace(**Sim.__dict__) 
    Sim_uniform.Name = 'CAD/Uniform'
    Sim_uniform.AsterFile = 'MB_steady_mech_uniform' # This file must be in Scripts/$SIMULATION/Aster
    Sim_uniform.Flux = 100000

    Main_parameters = Namespace(Sim = Sim_uniform)
    VirtualLab.Parameters(Main_parameters, None, RunSim=True)

    VirtualLab.Sim.SetPoolRun('PoolRun_uniform') # change the function which will be run
    VirtualLab.Sim(ShowRes=False)

if ERMESAnalysis:
    # copy Sim parameters and add additional values
    Sim_ERMES = Namespace(**Sim.__dict__)
    Sim_ERMES.Name = 'CAD/ERMES'
    
    #############
    ### ERMES ###
    #############
    Sim_ERMES.CoilType = 'Pancake'
    Sim_ERMES.CoilDisplacement = [0,0,0.0015]
    Sim_ERMES.Frequency = 1e4
    Sim_ERMES.VacuumShape = 'cube'

    #############
    ### Aster ###
    #############
    Sim_ERMES.AsterFile = 'MB_steady_mech' # This file must be in Scripts/$SIMULATION/Aster
    Sim_ERMES.Current = 1000
    Sim_ERMES.NbClusters = 100


    Main_parameters = Namespace(Sim = Sim_ERMES)
    VirtualLab.Parameters(Main_parameters, None, RunSim=True)

    VirtualLab.Sim.SetPoolRun('PoolRun') # change back to default as it may have been changed in uniform
    VirtualLab.Sim(ShowRes=False)

