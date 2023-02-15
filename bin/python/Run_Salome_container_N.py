# main function called from container to run Salome mecca
#!/usr/bin/env python3
import argparse
import dill
import types


from Scripts.Common.VLContainer import Container_Utils as Utils
from Scripts.Common.VLModules import VLModule

# Read arguments from command line
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--pkl_file", help = "VirtualLab parameter file", required=True)
parser.add_argument("-I", "--Container_ID", help = "unique integer id for container communication", required=True)
args = parser.parse_args()


with open(args.pkl_file,'rb') as handle:
    VirtualLab = dill.load(handle)

#VirtualLab.Settings(Cleanup=False)
VirtualLab.Container = args.Container_ID
VirtualLab.tcp_sock = Utils.create_tcp_socket()

for attrname in ['start_module2','get_args','heartbeat','filter_runs']:
    at = getattr(VLModule,attrname)
    at2 = types.MethodType( at, VirtualLab ) # make a class attribute in to an instance method
    setattr(VirtualLab,attrname,at2)

VirtualLab.Mesh.clsname = 'VLModule'
VirtualLab.start_module2()
args = VirtualLab.run_args

VirtualLab.Mesh(**args)

# this step ensures the heartbeat stops
Utils.Cont_Finished(VirtualLab.Container, VirtualLab.tcp_sock)

#VirtualLab.CP = types.MethodType(VLModule._Cleanup, VirtualLab)
#VirtualLab.CP(False)

