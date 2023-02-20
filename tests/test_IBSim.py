#!/bin/python3
import os
import sys
import shutil
sys.dont_write_bytecode=True
from subprocess import Popen

TestName='IBSim' # sub direcotry within TutorialsDir where test scripts are kept

def test_Task1(TutorialsDir,ParsedArgs):
    Run = Popen(['VirtualLab','-f','{}/{}/Task1_Run.py'.format(TutorialsDir,TestName),*ParsedArgs])
    err = Run.wait()
    assert err==0
