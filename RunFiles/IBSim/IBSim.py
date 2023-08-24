#!/usr/bin/env python3
import sys
import os
sys.dont_write_bytecode=True
from types import SimpleNamespace as Namespace

from Scripts.Common.VirtualLab import VLSetup
from Scripts.Common.tools.inp2Med import main as inp2med
from Scripts.VLPackages.Salome import API as Salome

MeshName = 'HIVE_Heli_exp_NR_25pc_Rescale_CoilVac_v5'
#MeshName = 'HIVE_Heli_exp_NR_25pc_Rescale_CoilVac_Void_03'

UniformAnalysis = True
ERMESAnalysis = False

# ====================================================================
# Setup VirtualLab & its settings

Simulation = 'HIVE'
Project = 'IBSim'

VirtualLab=VLSetup(
           Simulation,
           Project)

VirtualLab.Settings(Mode='h')
    
# parameters common to uniform and ermes
Sim = Namespace()
Sim.Mesh = '{}_CodeAster'.format(MeshName)
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

# ===================================================================
# run analysis using a uniform load

if UniformAnalysis:
    # copy Sim parameters and add additional values
    Sim_uniform = Namespace(**Sim.__dict__) 
    Sim_uniform.Name = '{}/Uniform'.format(MeshName)
    Sim_uniform.AsterFile = 'MB_steady_mech_uniform' # This file must be in Scripts/$SIMULATION/Aster
    Sim_uniform.Flux = 100000

    Main_parameters = Namespace(Sim = Sim_uniform)
    VirtualLab.Parameters(Main_parameters, None, RunSim=True)

    VirtualLab.Sim.SetPoolRun('PoolRun_uniform') # change the function which will be run
    VirtualLab.Sim(ShowRes=False)


if ERMESAnalysis:
    Sim_ERMES = Namespace(**Sim.__dict__)
    Sim_ERMES.Name = '{}/ERMES'.format(MeshName)

    #############
    ### ERMES ###
    #############
    Sim_ERMES.MeshERMES = '{}_ERMES'.format(MeshName)
    Sim_ERMES.Frequency = 1e4

    #############
    ### Aster ###
    #############
    Sim_ERMES.AsterFile = 'MB_steady_mech' # This file must be in Scripts/$SIMULATION/Aster
    Sim_ERMES.Current = 1000
    Sim_ERMES.NbClusters = 100


    Main_parameters = Namespace(Sim = Sim_ERMES)

    VirtualLab.Parameters(Main_parameters, None, RunSim=True)

    VirtualLab.Sim.SetPoolRun('PoolRun_IBSim') # change the function which will be run
    VirtualLab.Sim(ShowRes=False)


#                'Nodes':{'PipeInV1':12480,
#                        'PipeInV2':280406,
#                        'PipeOutV1':16338,
#                        'PipeOutV2':306949            
#                        }
#                }



