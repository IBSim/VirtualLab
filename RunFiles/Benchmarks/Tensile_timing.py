#!/usr/bin/env python3
import sys
import os
sys.dont_write_bytecode=True
from types import SimpleNamespace as Namespace
import time
import pickle

import numpy as np
import matplotlib.pyplot as plt

from Scripts.Common.VirtualLab import VLSetup
import Scripts.Common.VLFunctions as VLF

NbSim = 4
NbRepeat = 1
launcher = 'sequential'
NbJobs = 2
MakeMesh = False
RunSim = True
CreatePlot = False

# ====================================================================
# Update running parameters with any passed via the command line
parsed_kwargs= VLF.parsed_kwargs(sys.argv)
NbSim = parsed_kwargs.get('NbSim',NbSim)
NbRepeat = parsed_kwargs.get('NbRepeat',NbRepeat)
launcher = parsed_kwargs.get('launcher',launcher)
NbJobs = int(parsed_kwargs.get('NbJobs',NbJobs))
MakeMesh = parsed_kwargs.get('MakeMesh',MakeMesh)
RunSim = parsed_kwargs.get('RunSim',RunSim)
CreatePlot = parsed_kwargs.get('CreatePlot',CreatePlot)

# ====================================================================
# Update running parameters with any passed via the command line
if launcher in ('seq','sequential'):
    launcher = dirname = 'sequential' ; NbJobs = 1 
elif launcher == 'process':
    dirname = "{}_{}".format(launcher,NbJobs)
elif launcher.startswith('mpi'):
    NbNodes = os.environ.get('SLURM_NNODES',1) # may need to change this for different systems
    dirname = "{}_{}_{}".format(launcher,NbJobs,NbNodes)

if RunSim:
    print("{} simulations performed using the {} launcher with {} jobs".format(NbSim,launcher,NbJobs))

# ====================================================================
# Parameters for mesh and simulation

# mesh main
Mesh = Namespace()
Mesh.Name = 'Notch1' # name of mesh
Mesh.File = 'DogBone' # Salome python file used to create mesh.
# Geometric Parameters
Mesh.Thickness = 0.003
Mesh.HandleWidth = 0.024
Mesh.HandleLength = 0.024
Mesh.GaugeWidth = 0.012
Mesh.GaugeLength = 0.04
Mesh.TransRad = 0.012
Mesh.HoleCentre = (0.0,0.0)
Mesh.Rad_a = 0.0005
Mesh.Rad_b = 0.001
# Meshing Parameters
Mesh.Length1D = 0.001
Mesh.Length2D = 0.001
Mesh.Length3D = 0.001
Mesh.HoleSegmentN = 30

# sim main
Sim = Namespace()
Sim.Name = '' # Name under which the simulation results will be saved.
Sim.AsterFile = 'Tensile'
Sim.Mesh = 'Notch1' # The mesh used in the simulation.
Sim.Force = 1000000 # Force applied in force controlled analysis.
Sim.Displacement = 0.01 # Enforced displacement in displacement controlled analysis.
Sim.Materials = 'Copper' # Material specimen is made of. Properties can be found in the 'Materials' directory.

Main_parameters = Namespace(Mesh=Mesh, Sim = Sim)

# sim var
Sim = Namespace()
Sim.Force = [1000]*NbSim
Sim.Name = ["{}/Sim_{}".format(dirname,i) for i in range(NbSim)]
Var_parameters = Namespace(Sim = Sim)

# ====================================================================
# Setup VirtualLab

Simulation = 'Tensile'
Project = 'Benchmarking/Tensile_timing'

VirtualLab=VLSetup(
           Simulation,
           Project)

VirtualLab.Settings(
           Mode='h',
           Launcher=launcher,
           NbJobs=NbJobs)

VirtualLab.Parameters(Main_parameters,
                      Var_parameters,
                      RunMesh=MakeMesh,
                      RunSim=RunSim
                      )

# ====================================================================
# Create Mesh
VirtualLab.Mesh()


# ====================================================================
# Perform simulations
pkl_file = "{}/Tensile_timing_{}.pkl".format(VirtualLab.PROJECT_DIR, NbSim)
if RunSim:
    times = []
    for _ in range(NbRepeat):
        st = time.time()
        VirtualLab.Sim()
        t = time.time() - st
        times.append(t)
    print('Avg. time for {} {}: {:.4f} s'.format(launcher,NbJobs,np.mean(times)))

    # Add timings to file
    with open(pkl_file,'ab') as f:
        pickle.dump({dirname:[launcher,NbJobs,times]},f)

# ====================================================================
if CreatePlot:

    data_dict = {}
    with open(pkl_file,'rb') as f:
        while True:
            try:
                _data = pickle.load(f)
                data_dict.update(_data)
            except:
                EOFError
                break

    print(data_dict)

    data = {}
    for key, val in data_dict.items():
        launcher,NbJobs,times = val
        avgtime = np.mean(times)
        if launcher not in data: data[launcher] = []
        data[launcher].append([NbJobs,avgtime])

    seq = data.pop('sequential')
    seq_time = seq[0][1]
    
    for key,val in data.items():
        val.append([1,seq_time])
        val = np.array(val)
        val = val[np.argsort(val[:,0])]
        maxnb = val[-1,0]

        plt.figure()
        plt.plot([1,maxnb],[seq_time,seq_time/maxnb],linestyle='--',c='k')
        plt.plot(*val.T,linestyle='-',marker='o',c='k',)
        plt.xscale('log',basex=2)
        plt.yscale('log',basey=2)
        plt.xlabel('Nb parallel jobs')
        plt.ylabel('Time (s)')
        plt.savefig("{}/{}.png".format(VirtualLab.PROJECT_DIR,key,format),format='png',dpi=600)
        plt.close()

