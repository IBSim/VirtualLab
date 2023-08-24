#!/usr/bin/env python3
import sys
import os
sys.dont_write_bytecode=True
from types import SimpleNamespace as Namespace

from Scripts.Common.VirtualLab import VLSetup
from Scripts.Common.tools.inp2Med import main as inp2med
from Scripts.VLPackages.Salome import API as Salome

ConvertMesh = False
UpdateMesh = False


MeshName = 'HIVE_Heli_exp_NR_25pc_Rescale_CoilVac_v5'
Nodes = {'PipeInV1':12480,
        'PipeInV2':280406,
        'PipeOutV1':16338,
        'PipeOutV2':306949            
        }
        
#MeshName = 'HIVE_Heli_exp_NR_25pc_Rescale_CoilVac_Void_03'
#Nodes = {'PipeInV1':20260,
#        'PipeInV2':287942,
#        'PipeOutV1':25109,
#        'PipeOutV2':280816            
#        }


# ====================================================================
# Setup VirtualLab & its settings

Simulation = 'HIVE'
Project = 'IBSim'

VirtualLab=VLSetup(
           Simulation,
           Project)

VirtualLab.Settings(Mode='h')
    
# ===================================================================
# convert mesh
if ConvertMesh:
    inp_mesh = "{}/{}.inp".format(VirtualLab.MESH_DIR,MeshName) # inp mesh must be placed in mesh directory
    inp2med(inp_mesh)

# ===================================================================
# make changes to mesh for ERMES & code aster compatibility
if UpdateMesh:
    script = "{}/IBSim.py".format(VirtualLab.SIM_MESH)
    DataDict = {'Mesh':"{}/{}.med".format(VirtualLab.MESH_DIR,MeshName),
                'Nodes':Nodes
                }
    err = Salome.Run(script,DataDict=DataDict)






