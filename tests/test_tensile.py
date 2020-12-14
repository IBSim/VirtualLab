#!/bin/python3
import os
import sys
import shutil
sys.dont_write_bytecode=True

dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,dirname)
import VLconfig
sys.path.pop(0)

Name='Tensile'

tmpInput = '/tmp/unit/Input'
tmpOutput = '/tmp/unit/Output'
tmpTemp = '/tmp/unit/Temp'

sys.path.insert(0,"{}/RunFiles/Tutorials/{}".format(dirname,Name))
tmpdirs = ['INPUT_DIR={}'.format(tmpInput),'TEMP_DIR={}'.format(tmpTemp),'OUTPUT_DIR={}'.format(tmpOutput)]
chargs = ['Mode=T','ShowRes=False','ShowMesh=False']
shutil.copytree("{}/{}".format(VLconfig.InputDir,Name),"{}/{}".format(tmpInput,Name))

def test_Task1():
	sys.argv = [None] + chargs + tmpdirs
	import Task1_Run

def test_Task2():
	sys.argv = [None] + chargs + tmpdirs
	import Task2_Run

def test_Task3():
	sys.argv = [None] + chargs + tmpdirs
	import Task3_Run

def test_Task4():
	sys.argv = [None] + chargs + tmpdirs
	import Task4_Run

def test_Cleanup():
	shutil.rmtree('/tmp/unit')
