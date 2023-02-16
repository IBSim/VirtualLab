# main function called from container to run Salome mecca
#!/usr/bin/env python3
import argparse
import dill
import types

from Scripts.Common.VLContainer import Container_Utils as Utils
from Scripts.Common.VLModules import VLModule2

# Read arguments from command line
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--pkl_file", help = "VirtualLab parameter file", required=True)
parser.add_argument("-I", "--Container_ID", help = "unique integer id for container communication", required=True)
args = parser.parse_args()

with open(args.pkl_file,'rb') as handle:
    VL = dill.load(handle)
VL.Container = args.Container_ID

Module_inst = VLModule2(VL)

Module_inst.Run()

Module_inst.Terminate()


