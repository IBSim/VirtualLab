"""
conftest.py
"""
import os

import pytest

def pytest_addoption(parser):
    parser.addoption("--VLargs", action="store", default="")

@pytest.fixture(scope="session")
def TutorialsDir(pytestconfig):
    # maybe needs to be more robust
    VLdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tutorials_path = "{}/RunFiles/Tutorials".format(VLdir)
    return tutorials_path

@pytest.fixture(scope="session")
def ParsedArgs(pytestconfig):
    global_args = {'Mode':'T','ShowMesh':False,'ShowRes':False}

    cmdargs = pytestconfig.getoption("VLargs")
    if cmdargs:
        for arg in cmdargs.split(' '):
            if arg=='-K': continue
            split = arg.split('=')
            if len(split)==2:
                varname,value = split
                global_args[varname] = value
            else:
                # TODO add error for this
                print("Unknown arg {}".format(arg))

    cmd_list = ["{}={}".format(key,val) for key,val in global_args.items()]
    cmd_list.insert(0,'-K')
    return cmd_list
