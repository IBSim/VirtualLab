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

# flags 
MakeMesh = False
RunSim = True
CreatePlot = False
# parameters relating to benchmark 
Launcher = 'process'
NbSim = 2
NbRepeat = 1


# ====================================================================
# Update running parameters with any passed via the command line
parsed_kwargs= VLF.parsed_kwargs(sys.argv)
MakeMesh = parsed_kwargs.get('MakeMesh',MakeMesh)
RunSim = parsed_kwargs.get('RunSim',RunSim)
CreatePlot = parsed_kwargs.get('CreatePlot',CreatePlot)
Launcher = parsed_kwargs.get('Launcher',Launcher)
NbSim = parsed_kwargs.get('NbSim',NbSim)
NbRepeat = parsed_kwargs.get('NbRepeat',NbRepeat)

# ====================================================================
# Update running parameters with any passed via the command line
if Launcher in ('seq','sequential'):
    Launcher = dirname = 'sequential' ; NbJobs = 1; NbSim = 1
elif Launcher == 'process':
    NbJobs = NbSim
    dirname = "{}_{}".format(Launcher,NbSim)
elif Launcher.startswith(('mpi','srun')):
    if Launcher in ('mpi_worker','srun_worker'): NbJobs = NbSim
    else: NbJobs = NbSim+1
    NbNodes = os.environ.get('SLURM_NNODES',1) # may need to change this for different systems
    dirname = "{}_{}_{}".format(Launcher,NbNodes,NbSim)

# ====================================================================
# Parameters for mesh and simulation

# mesh main
Mesh = Namespace()
Mesh.Name = 'NoVoid'
Mesh.File = 'Disc'
Mesh.Radius = 0.0063
Mesh.HeightB = 0.00125
Mesh.HeightT = 0.00125
Mesh.Length1D = 0.00025
Mesh.Length2D = 0.00025
Mesh.Length3D = 0.00025

# sim main
Sim = Namespace()
Sim.Name = 'Single'
Sim.AsterFile = 'Disc_Lin'
Sim.Mesh = 'NoVoid'
Sim.Model = '3D'
Sim.Solver = 'MUMPS'
Sim.Energy = 5.32468714
Sim.LaserT= 'Trim'
Sim.LaserS = 'Gauss'
Sim.BottomHTC = 0
Sim.TopHTC = 0
Sim.ExtTemp = 20
Sim.InitTemp = 20
Sim.Materials = {'Top':'Copper', 'Bottom':'Copper'}
Sim.dt = [(0.00002,100,2), (0.00025,200,4)]
Sim.Theta = 0.5

Main_parameters = Namespace(Mesh=Mesh, Sim = Sim)

# sim var
Sim = Namespace()
Sim.Name = ["{}/Sim_{}".format(dirname,i) for i in range(NbSim)]

Var_parameters = Namespace(Sim = Sim)


# ====================================================================
# Setup VirtualLab

Simulation = 'LFA'
Project = 'Benchmarking/LFA_timing'

VirtualLab=VLSetup(
           Simulation,
           Project)

VirtualLab.Settings(
           Mode='h',
           Launcher=Launcher,
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
pkl_file = "{}/timing.pkl".format(VirtualLab.PROJECT_DIR)
if RunSim:
    print("{} simulations performed using the {} launcher".format(NbSim,Launcher))
    times = []
    for _ in range(NbRepeat):
        st = time.time()
        VirtualLab.Sim()
        t = time.time() - st
        times.append(t)
    print('Avg. time for {} to perform {}: {:.4f} s'.format(Launcher,NbSim,np.mean(times)))

    # Add timings to file
    data = [Launcher,NbSim,times]
    if Launcher.startswith(('mpi','srun')): data.append(NbNodes)
    with open(pkl_file,'ab') as f:
        pickle.dump({dirname:data},f)

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

    data = {}
    for key, val in data_dict.items():
        Launcher_name = val[0]
        if Launcher_name.startswith(('mpi','srun')):
            Launcher_name += f"_{val[3]}" # add number of nodes
        if Launcher_name not in data: data[Launcher_name] = []

        NbSim,times = val[1],val[2]
        avgtime = np.mean(times)

        data[Launcher_name].append([NbSim,avgtime])

    seq = data.pop('sequential')
    seq_time = seq[0][1]

    f, ax = plt.subplots()
    for key,val in data.items():

        NbSims,times = np.array(list(zip(*val)))
        speed_up = seq_time*NbSims/times
        ax.scatter(NbSims,speed_up, label=key)

    ax.legend()
    xlim_upper = ax.get_xlim()[1]
    ax.plot([1,xlim_upper],[1,xlim_upper],linestyle='-',c='k',label='Perfect scaling')
    ax.set_xlim(left=0.5)
    ax.set_ylim(bottom=0.5)
    ax.set_xlabel("No. parallel simulations")
    ax.set_ylabel("Speed up")
    f.savefig("{}/Timing.png".format(VirtualLab.PROJECT_DIR),format='png',dpi=600)


