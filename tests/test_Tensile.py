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

Name='Tensile'

TutorialsDir = "{}/RunFiles/Tutorials/{}".format(VLdir,Name)
OutputDir = '{}/VLTutorial_{}'.format(VLconfig.TEMP_DIR,Name)
ParsedArgs = '-k Mode=T -k ShowMesh=False -k ShowRes=False -k OutputDir={}'.format(OutputDir)

def test_Task1():
    # Run = Popen('VirtualLab -f {}/Task1_Run.py {}'.format(TutorialsDir,ParsedArgs),shell='TRUE')
    Run = Popen(['VirtualLab','-f','{}/Task1_Run.py'.format(TutorialsDir),ParsedArgs])
    err = Run.wait()
    assert err==0

def test_Task2():
    Run = Popen(['VirtualLab','-f','{}/Task2_Run.py'.format(TutorialsDir),ParsedArgs])
    err = Run.wait()
    assert err==0

def test_Task3():
    Run = Popen(['VirtualLab','-f','{}/Task3_Run.py'.format(TutorialsDir),ParsedArgs])
    err = Run.wait()
    assert err==0

def test_Task4():
    Run = Popen(['VirtualLab','-f','{}/Task4_Run.py'.format(TutorialsDir),ParsedArgs])
    err = Run.wait()
    assert err==0
