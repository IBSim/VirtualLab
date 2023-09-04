#!/usr/bin/env python3
import sys
sys.dont_write_bytecode=True
from types import SimpleNamespace as Namespace
from Scripts.Common.VirtualLab import VLSetup

# a few tests to test out parameter handling
def test_master_no_name():
    VirtualLab = VLSetup('Unit','junk')
    # Parameters used to generate mesh
    Mesh = Namespace() # create mesh namespace
    VirtualLab.Parameters(Namespace(Mesh=Mesh))

def test_var_no_name():
    VirtualLab = VLSetup('Unit','junk')
    # Parameters used to generate mesh
    Mesh = Namespace() # create mesh namespace
    Mesh.Name = 'bob'
    Mesh2 = Namespace()
    VirtualLab.Parameters(Namespace(Mesh=Mesh),Namespace(Mesh=Mesh2))

def test_no_name_skip():
    # should skip over this as RunMesh is False
    VirtualLab = VLSetup('Unit','junk')
    # Parameters used to generate mesh
    Mesh = Namespace() # create mesh namespace
    VirtualLab.Parameters(Namespace(Mesh=Mesh),RunMesh=False)


def test_no_name():
    VirtualLab = VLSetup('Unit','junk')
    # Parameters used to generate mesh
    Mesh = Namespace() # create mesh namespace
    Mesh2 = Namespace()
    Mesh2.Name = []
    VirtualLab.Parameters(Namespace(Mesh=Mesh),Namespace(Mesh=Mesh2))

def test_diff_length():
    VirtualLab = VLSetup('Unit','junk')
    # Parameters used to generate mesh
    Mesh = Namespace() # create mesh namespace
    Mesh2 = Namespace()
    Mesh2.Name = [1,2,3]
    Mesh2.variable = ['a','b']
    VirtualLab.Parameters(Namespace(Mesh=Mesh),Namespace(Mesh=Mesh2))

if __name__=='__main__':
    test_no_name_skip()