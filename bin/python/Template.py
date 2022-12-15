#!/usr/bin/env python3
# Template for python script to run analysis
import argparse
import os
# Read arguments from command line
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--Parameters_Master", help = "VirtualLab parameter file", required=True)
parser.add_argument("-v", "--Parameters_Var", help = "VirtualLab parameter file", default=None)
parser.add_argument("-p", "--Project", help = "Main Directory for project data", required=True)
parser.add_argument("-s", "--Simulation", help = "Sub-Directory for simulation data", required=True)
parser.add_argument("-I", "--Container_ID", help = "unique integer id for container communication", required=True)
args = parser.parse_args()
Cont_id=args.Container_ID
os.chdir('/home/ibsim/VirtualLab')
# import and setup the VLModule class
from Scripts.Common.VLModules import VLModule
VirtualLab=VLModule(
           args.Simulation,
           args.Project,
           Cont_id)
# get parameters which have been passed over the network from the master.
params_dict = VirtualLab.get_method_params()
VirtualLab.Parameters(
           args.Parameters_Master,
           args.Parameters_Var,
           RunSim=True,
           RunMesh=True)
# get the method arguments, again these should have already 
# been communicated over
kwargs = VirtualLab.get_method_args()
# call method to run analysis
VirtualLab.#MethodName(**kwargs)
