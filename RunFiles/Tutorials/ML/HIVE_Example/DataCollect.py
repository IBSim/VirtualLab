#!/usr/bin/env python3
import sys
sys.dont_write_bytecode=True
from types import SimpleNamespace as Namespace
import numpy as np
from Scripts.Common.VirtualLab import VLSetup
from Scripts.Common.tools import Sampling

# ====================================================================
# flags
CreateMesh = True
RunSimulations = True
ExtractData = True

DispX = [-0.005,0.005]
DispY = [-0.015,0.015]
DispZ = [0.003,0.006]
RotationY = [-5,5]
CoolantT = [30,80]
CoolantV = [0.65784,7.23626]
Current = [200,2000]
PS = [DispX,DispY,DispZ,RotationY,CoolantT,CoolantV,Current]

NTrain = 700 # will be collected using the sobol samplign scheme
NTest = 300 # samples randomly drawn from across the parameter space

CoilType='Pancake' # this can be 'Pancake' or 'HIVE'

# ====================================================================

# Setup VirtualLab
VirtualLab=VLSetup('HIVE','ML_analysis')

# ====================================================================
# Create mesh
# ====================================================================

# Parameters used to generate mesh
Mesh = Namespace() # create mesh namespace
Mesh.Name = 'HIVE_component'
Mesh.File = 'Monoblock' 
# Geometrical Dimensions for component
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
# Mesh parameters
Mesh.Length1D = 0.003
Mesh.Length2D = 0.003
Mesh.Length3D = 0.003
Mesh.SubTile = [0.001, 0.001, 0.001]
Mesh.CoilFace = 0.0006
Mesh.Fillet = 0.0005
Mesh.Deflection = 0.01
Mesh.PipeSegmentN = 24 # Number of segments for pipe circumference

VirtualLab.Parameters(Namespace(Mesh=Mesh),
                      RunMesh=CreateMesh,
                      )
# run mesh
VirtualLab.Mesh()

# ====================================================================
# Run simulation
# ====================================================================

# Parameters used to generate simulation data

# main parameters
Sim = Namespace()
Sim.Name = ''
Sim.Mesh = 'HIVE_component' # The mesh used for the simulation
Sim.Frequency = 101.3e3
Sim.CoilType = CoilType
Sim.Materials = {'Block':'Tungsten_NL', 'Pipe':'Tungsten_NL', 'Tile':'Tungsten_NL'}
Sim.AsterFile = 'MB_steady_mech'
Sim.Model = '3D'
Sim.Solver = 'MUMPS'
Sim.NbClusters = 500
Sim.Pipe = {'Type':'smooth tube', 'Diameter':0.0127, 'Length':0.2}
Sim.BlockPipeTCC=None
Sim.TileBlockTCC=None
Sim.BlockEmissivity=0.34
Sim.TileEmissivity=0.34
Sim.TempExt = 23

Parameters_main = Namespace(Sim = Sim)

# parametric values
Sim = Namespace(CoilDisplacement=[],CoilRotation=[],Coolant=[],Current=[])
for sample_method, nb, seed in [['Sobol',NTrain,123],['Random',NTest,100]]:
    Sampler = Sampling(sample_method,range=PS,bounds=False,seed=seed) # seed only used for random sampling
    Samples = Sampler.get(nb)

    CoilDisplacement = np.vstack((Samples[:3])).T
    Sim.CoilDisplacement += CoilDisplacement.tolist()
    CoilRotation = np.zeros((nb,3))
    CoilRotation[:,1] = Samples[3] # only rotating along y axis
    Sim.CoilRotation += CoilRotation.tolist()
    Coolant = [{'Temperature': t, 'Pressure': 0.5, 'Velocity': v} for t,v in zip(*Samples[4:6])]
    Sim.Coolant += Coolant
    Sim.Current += Samples[6]

Name_train = ["{}_coil/Train/Sim_{}".format(CoilType,i) for i in range(NTrain)]
Name_test = ["{}_coil/Test/Sim_{}".format(CoilType,i) for i in range(NTest)]
Sim.Name = Name_train + Name_test

Parameters_var = Namespace(Sim = Sim)

VirtualLab.Parameters(Parameters_main,
                      Parameters_var,
                      RunSim=RunSimulations,
                      )

# change these depending on your system
VirtualLab.Settings(Launcher='process', NbJobs=5) 
# run simulations
VirtualLab.Sim(RunCoolant=True, RunERMES=True, RunAster=True)


# ====================================================================
# Extract data from simulation and put in hdf files
# ====================================================================
VirtualLab.Settings(Launcher='sequential')

# get values for the power and variation on the surface 'CoilFace' and add to file PowerVariation.hdf
DA = Namespace()
DA.Name = ''
DA.File = ('CollectData','CompileData')
DA.DataFile = '{}_coil/PowerVariation.hdf'.format(CoilType) # File data will be stored to
DA.Collect = [{'Name':'Features',
            'Function':'Inputs',
            'args':[["CoilDisplacement [0]", "CoilDisplacement [1]",
                        "CoilDisplacement [2]", "CoilRotation [1]"]]
                },
            {'Name':'Power',
                'Function':'Power_ERMES',
                'args':["PreAster/ERMES.rmed"]
                },
            {'Name':'Variation',
                'Function':'Variation_ERMES',
                'args':["PreAster/ERMES.rmed","CoilFace"],
                },
            ]
Parameters_main = Namespace(DA=DA)

DA = Namespace(Name=[],Group=[],CompileData=[])
for dname in ['Train','Test']:
    DA.Group.append(dname)# group in the hdf file which the data will be saved to
    DA.CompileData.append('{}_coil/{}'.format(CoilType,dname) )# Results directory which will be iterated over
    DA.Name.append('_DataCollect/{}'.format(dname)) # analysis name
Parameters_var = Namespace(DA=DA)

VirtualLab.Parameters(Parameters_main,
                      Parameters_var,
                      RunDA=ExtractData)
VirtualLab.DA()

# ====================================================================
# get values for the Joule heating on surface CoilFace and add to JouleHeating.hdf
DA = Namespace()
DA.Name = ''
DA.File = ('CollectData','CompileData')
DA.DataFile = '{}_coil/JouleHeating.hdf'.format(CoilType) # File data will be stored to
DA.Collect = [{'Name':'Features',
            'Function':'Inputs',
            'args':[["CoilDisplacement [0]", "CoilDisplacement [1]",
                        "CoilDisplacement [2]", "CoilRotation [1]"]]
                },
            {'Name':'SurfaceJH',
                'Function':'NodalMED',
                'args':["PreAster/ERMES.rmed","Joule_heating"],
                'kwargs':{'GroupName':'CoilFace'}
                },
            ]
Parameters_main = Namespace(DA=DA)
# iterating over the ame two directories so parameters_var is the same as previously defined

VirtualLab.Parameters(Parameters_main,
                      Parameters_var,
                      RunDA=ExtractData)
VirtualLab.DA()

# ====================================================================
# get temperature nodal values throughout component and write to TempNodal.hdf
DA = Namespace()
DA.Name = ''
DA.File = ('CollectData','CompileData')
DA.DataFile = '{}_coil/TempNodal.hdf'.format(CoilType) # File data will be stored to
DA.Collect = [{'Name':'Features',
            'Function':'Inputs',
            'args':[["CoilDisplacement [0]", "CoilDisplacement [1]",
                    "CoilDisplacement [2]", "CoilRotation [1]",
                    "Coolant['Temperature']","Coolant['Velocity']",
                    "Current"]]   
            },                   
            {'Name':'Temperature',
            'Function':'NodalMED',
            'args':["Aster/Thermal.rmed","Temperature"]
            },   
            ]
Parameters_main = Namespace(DA=DA)
# iterating over the ame two directories so parameters_var is the same as previously defined

VirtualLab.Parameters(Parameters_main,
                      Parameters_var,
                      RunDA=ExtractData)
VirtualLab.DA()

# ====================================================================
# get Von Mises nodal values throughout component and write to 
DA = Namespace()
DA.Name = ''
DA.File = ('CollectData','CompileData')
DA.DataFile = '{}_coil/VMNodal.hdf'.format(CoilType) # File data will be stored to
DA.Collect = [{'Name':'Features',
            'Function':'Inputs',
            'args':[["CoilDisplacement [0]", "CoilDisplacement [1]",
                    "CoilDisplacement [2]", "CoilRotation [1]",
                    "Coolant['Temperature']","Coolant['Velocity']",
                    "Current"]]   
            },                   
            {'Name':'VonMises',
            'Function':'NodalMED',
            'args':["Aster/Thermal.rmed","Stress_Eq_Nd"],
            'kwargs':{ 'ComponentName':"VMIS"}
            },                    
            ]
Parameters_main = Namespace(DA=DA)
# iterating over the ame two directories so parameters_var is the same as previously defined

VirtualLab.Parameters(Parameters_main,
                      Parameters_var,
                      RunDA=ExtractData)
VirtualLab.DA()

