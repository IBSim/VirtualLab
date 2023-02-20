#!/bin/python3
import os
import sys
import shutil
sys.dont_write_bytecode=True
from subprocess import Popen

TestName='Tensile' # sub direcotry within TutorialsDir where test scripts are kept

def test_Task1(TutorialsDir,ParsedArgs):
    Run = Popen(['VirtualLab','-f','{}/{}/Task1_Run.py'.format(TutorialsDir,TestName),*ParsedArgs])
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

def test_Task3a(TutorialsDir,ParsedArgs):
    Run = Popen(['VirtualLab','-f','{}/{}/Task3_Run_a.py'.format(TutorialsDir,TestName),*ParsedArgs])
    err = Run.wait()
    assert err==0

def test_Task3b(TutorialsDir,ParsedArgs):
    Run = Popen(['VirtualLab','-f','{}/{}/Task3_Run_b.py'.format(TutorialsDir,TestName),*ParsedArgs])
    err = Run.wait()
    assert err==0

def test_Task3c(TutorialsDir,ParsedArgs):
    Run = Popen(['VirtualLab','-f','{}/{}/Task3_Run_c.py'.format(TutorialsDir,TestName),*ParsedArgs])
    err = Run.wait()
    assert err==0
            
def test_Task4(TutorialsDir,ParsedArgs):
    Run = Popen(['VirtualLab','-f','{}/{}/Task4_Run.py'.format(TutorialsDir,TestName),*ParsedArgs])
    err = Run.wait()
    assert err==0
    
    
