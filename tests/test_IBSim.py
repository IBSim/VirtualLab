#!/bin/python3
import os
import sys
import shutil
sys.dont_write_bytecode=True
from subprocess import Popen

VLdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,VLdir)
import VLconfig
sys.path.pop(0)

Name='IBSim'

TutorialsDir = "{}/RunFiles/Tutorials/{}".format(VLdir,Name)
ParsedArgs = '-K Mode=T -K ShowMesh=False -K ShowRes=False'

def test_Task1():
    Run = Popen(['VirtualLab','-f','{}/Task1_Run.py'.format(TutorialsDir),ParsedArgs])
    err = Run.wait()
    assert err==0


