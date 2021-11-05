#!/usr/bin/env python3

import sys
sys.dont_write_bytecode=True
import os
from subprocess import Popen, PIPE, STDOUT
import uuid
import pickle

import VLconfig

Exec = getattr(VLconfig,'SalomeExec','salome')
Dir = os.path.dirname(os.path.abspath(__file__))

Container = getattr(VLconfig,'SalomeContainer',None)
if Container:
    import ContainerConfig
    SalomeContainer = getattr(ContainerConfig,Container)

def Kill(Port):
	if type(Port) == int:
		KillPort=Port
	elif type(Port) == str:
		with open(Port,'r') as f:
			KillPort = int(f.readline())
	else : KillPort = 0

	if False:
		cmlst = Exec.split() + ['kill', str(KillPort)]
		SubProc = Popen(cmlst)
	else :
		if Container:
			command = "{} {} kill {}".format(SalomeContainer.Call, SalomeContainer.SalomeExec,KillPort)
		else:
			command = "{} kill {}".format(Exec, KillPort)
		SubProc = Popen(command, shell='TRUE')
	SubProc.wait()

def Run(Script, AddPath = [], DataDict = {}, OutFile=None, GUI=False, tempdir = '/tmp'):
	'''
	AddPath: Additional paths that Salome will be able to import from
	DataDict: a dictionary of the arguments that Salome will get
	OutFile: The log file you want to write stdout to
	GUI: Opens a new instance with GUI (useful for testing)
	tempdir: Location where pickled object can be written to
	'''

	# Add paths provided to python path for subprocess
	AddPath = [AddPath] if type(AddPath) == str else AddPath
	PyPath = ["{}:".format(path) for path in AddPath + [Dir]]
	PyPath = "".join(PyPath)

	_argstr = []
	if DataDict:
		pth = "{}/DataDict_{}.pkl".format(tempdir,uuid.uuid4())
		with open(pth,'wb') as f:
			pickle.dump(DataDict,f)
		_argstr.append('DataDict={}'.format(pth))
	argstr = ",".join(_argstr)


	portfile = "{}/{}".format(tempdir,uuid.uuid4())
	GUIflag = '-g' if GUI else '-t'
	env = {**os.environ, 'PYTHONPATH': PyPath + os.environ.get('PYTHONPATH','')}

	# Run mesh in Salome
	if Container:
		command = "{} {} {} --ns-port-log {} {} args:{} ".format(SalomeContainer.Call,
														  SalomeContainer.SalomeExec,
														  GUIflag, portfile, Script, argstr)
	else:
		command = "{} {} --ns-port-log {} {} args:{} ".format(Exec, GUIflag, portfile, Script, argstr)
	if OutFile:
		with open(OutFile,'w') as f:
			SubProc = Popen(command, shell='TRUE',cwd=tempdir, stdout=f, stderr=f,env=env)
	else :
		SubProc = Popen(command, shell='TRUE',cwd=tempdir, stdout=sys.stdout, stderr=sys.stderr,env=env)

	ReturnCode = SubProc.wait()

	Kill(portfile)

	return ReturnCode
