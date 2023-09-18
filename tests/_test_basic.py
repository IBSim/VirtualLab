#!/bin/python3
import os
import sys
import shutil
sys.dont_write_bytecode=True
from subprocess import Popen

VLdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,VLdir)
# import VLconfig
sys.path.pop(0)

TestName='basic' # sub direcotry within TutorialsDir where test scripts are kept

def test_Task1(TutorialsDir,ParsedArgs):
    Run = Popen(['VirtualLab','-f','{}/{}/Task1_Run.py'.format(TutorialsDir,TestName),*ParsedArgs])
    err = Run.wait()
    assert err==0

def setup_function(function):
    print("setting up", function)

def test_func1():
    assert True

# def test_func2():
#     assert False
