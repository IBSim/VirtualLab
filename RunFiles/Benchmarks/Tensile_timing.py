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

NbSim = 2
NbRepeat = 2
Launcher = 'process'
MakeMesh = False
RunSim = True
CreatePlot = False

# ====================================================================
# Update running parameters with any passed via the command line
parsed_kwargs= VLF.parsed_kwargs(sys.argv)
NbSim = parsed_kwargs.get('NbSim',NbSim)
NbRepeat = parsed_kwargs.get('NbRepeat',NbRepeat)
Launcher = parsed_kwargs.get('Launcher',Launcher)
MakeMesh = parsed_kwargs.get('MakeMesh',MakeMesh)
RunSim = parsed_kwargs.get('RunSim',RunSim)
CreatePlot = parsed_kwargs.get('CreatePlot',CreatePlot)

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
pkl_file = "{}/Tensile_timing.pkl".format(VirtualLab.PROJECT_DIR)
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
        print('here')
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

        x,y = list(zip(*val))
        y_ratio = np.array(y)/seq_time
        ax.scatter(x,y_ratio, label=key)

    ax.legend()
    xlims = ax.get_xlim()
    ax.plot(xlims,[1,1],linestyle='--',c='k')
    ax.set_xlim(*xlims)
    ax.set_ylim(bottom=0.75)
    ax.set_xlabel("No. parallel simulations")
    ax.set_ylabel("Scaling ratio")
    f.savefig("{}/Timing.png".format(VirtualLab.PROJECT_DIR),format='png',dpi=600)


