# main function called from container to run CodeAster/Ermes
#!/usr/bin/env python3
import argparse
import os
# Read arguments from command line
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--Parameters_Master", help = "VirtualLab parameter file", required=True)
parser.add_argument("-v", "--Parameters_Var", help = "VirtualLab parameter file", default=None)
parser.add_argument("-p", "--Project", help = "Main Directory for project data", required=True)
parser.add_argument("-s", "--Simulation", help = "Sub-Directory for simulation data", required=True)
parser.add_argument("-I", "--Container_ID", help = "unique integer id for container comunication", required=True)
args = parser.parse_args()
Cont_id=args.Container_ID
os.chdir('/home/ibsim/VirtualLab')
from Scripts.Common.VLModules import VL_Module
VirtualLab=VL_Module(
           args.Simulation,
           args.Project,
           Cont_id)

VirtualLab.Settings(Mode='Interactive')

VirtualLab.Parameters(
           args.Parameters_Master,
           args.Parameters_Var,
           RunSim=True,
           RunMesh=True)

VirtualLab.Sim(Module=True)
