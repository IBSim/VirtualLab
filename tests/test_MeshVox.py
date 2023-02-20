#!/bin/python3
import os
import sys
import shutil
sys.dont_write_bytecode=True
from subprocess import Popen

TestName='Mesh_Voxelisation' # sub direcotry within TutorialsDir where test scripts are kept

def test_Task1(TutorialsDir,ParsedArgs):
    Run = Popen(['VirtualLab','-f','{}/{}/Task1_Run.py'.format(TutorialsDir,TestName),*ParsedArgs])
    err = Run.wait()
    assert err==0

def test_Task2_pre(TutorialsDir,ParsedArgs):
    Run = Popen(['VirtualLab','-f','{}/Task2_Pre-setup_Run.py'.format(TutorialsDir),*ParsedArgs])
    err = Run.wait()
    assert err==0

def test_Task2(TutorialsDir,ParsedArgs):
    Run = Popen(['VirtualLab','-f','{}/{}/Task2_Run.py'.format(TutorialsDir,TestName),*ParsedArgs])
    err = Run.wait()
    assert err==0

def test_Task3(TutorialsDir,ParsedArgs):
    Run = Popen(['VirtualLab','-f','{}/{}/Task3_Run.py'.format(TutorialsDir,TestName),*ParsedArgs])
    err = Run.wait()
    assert err==0
