#!/usr/bin/env python3

import sys
sys.dont_write_bytecode=True
import os
from subprocess import Popen, PIPE, STDOUT
import uuid
import VLconfig

__all__ = ['Salome']

class Salome():
	def __init__(self, VL, **kwargs):
		self.cwd = getattr(VL, 'TEMP_DIR', os.getcwd())

		self.Logger = VL.Logger

		# How to call salome (can be changed for different versions etc.)
		self.Exec = getattr(VLconfig,'SalomeExec','salome')

		# AddPath will always add these paths to salome environment
		self.AddPath = kwargs.get('AddPath',[]) + [os.path.dirname(os.path.realpath(__file__))]
		self.Ports = []


#	def WriteArgs(self,ArgDict):
#		# Args = []
#		# for key, value in ArgDict.items()
#		# 	Args.append("{}={}".format(key, value))

#		for key, value in ArgDict.items()

#	def __ArgEncode__(self,val):
#		tp = type(val)
#		if tp == int:
#			return "__int{}".format(val)
#		elif tp == float:
#			return "float{}".format(val)
#		elif tp == str:
#			return "__str{}".format(val)
#		elif tp == bool:
#			return "_bool{}".format(val)
#		elif val == None:
#			return "_none{}".format(val)
#		elif tp == list:
#			lstring = "_list"
#			for i, lval in enumerate(val):
#				self.__ArgEncode__(lval)
#				string = "^{:04}^".format()
#			lstring+=

	def Start(self, Num=1,**kwargs):
		# If only OutFile is provided as a kwarg then ErrFile is set to this also
		OutFile = ErrFile = kwargs.get('OutFile', None)
		ErrFile = kwargs.get('ErrFile',ErrFile)

		output = ''
		if OutFile: output += " >>{}".format(OutFile)
		if ErrFile: output += " 2>>{}".format(ErrFile)

		self.Logger("Initiating Salome\n", Print=True)

		SalomeSP = []
		NewPorts = []
		for i in range(Num):
			portfile = "{}/{}".format(self.cwd,uuid.uuid4())
			SubProc = Popen('cd {};{} -t --ns-port-log {} {}'.format(self.cwd, self.Exec, portfile, output), shell='TRUE')
			SalomeSP.append((SubProc,portfile))

		for SubProc, portfile in SalomeSP:
			if not self.Success(SubProc):
				self.Logger("Error during Salome initiation",Print=True)
				continue

			with open(portfile,'r') as f:
				port = int(f.readline())
			NewPorts.append(port)

		if NewPorts:
			self.Logger('{} new Salome sessions opened on port(s) {}\n'.format(len(NewPorts),NewPorts))
		else:
			self.Logger('No new Salome sessions initiated\n')
		self.Ports.extend(NewPorts)

		return NewPorts

	def Shell(self, Script, **kwargs):
		'''
		kwargs available:
		OutFile: The log file you want to write stdout to (default is /dev/null)
		ErrFile: The log file you want to write stderr to (default is OutLog)
		AddPath: Additional paths that Salome will be able to import from
		ArgDict: a dictionary of the arguments that Salome will get
		ArgList: a list of arguments to be passed to Salome
		GUI: Opens a new instance with GUI (useful for testing)
		'''
		AddPath = kwargs.get('AddPath',[])
		ArgDict = kwargs.get('ArgDict', {})
		ArgList = kwargs.get('ArgList',[])

		# Add paths provided to python path for subprocess (self.COM_SCRIPTS and self.SIM_SCRIPTS is always added to path)
		AddPath = [AddPath] if type(AddPath) == str else AddPath
		PythonPath = ["{}:".format(path) for path in AddPath+self.AddPath]
		PythonPath = ["PYTHONPATH="] + PythonPath + ["$PYTHONPATH;export PYTHONPATH;"]
		PythonPath = "".join(PythonPath)

		# Write ArgDict and ArgList in format to pass to salome
		Args = ["{}={}".format(key, value) for key, value in ArgDict.items()]
		Args = ",".join(ArgList + Args)

		if kwargs.get('GUI',False):
			command = "{} {} args:{}".format(self.Exec, Script, Args)
			SubProc = Popen(PythonPath + command, shell='TRUE')
			return SubProc

		if not self.Ports:
			self.Start()

		Port = kwargs.get('Port', self.Ports[0])

		OutFile = ErrFile = kwargs.get('OutFile', None)
		ErrFile = kwargs.get('ErrFile',ErrFile)

		output = ''
		if OutFile: output += " >>{}".format(OutFile)
		if ErrFile: output += " 2>>{}".format(ErrFile)

		command = "{} shell -p{!s} {} args:{} {}".format(self.Exec,Port, Script, Args, output)

		SubProc = Popen(PythonPath + command, shell='TRUE')
		return SubProc

	def Success(self,SubProc):
		# If it hasn't finished it waits
		if SubProc.poll() == None:
			SubProc.wait()
		return SubProc.returncode == 0

	def Close(self, Ports):
		if type(Ports) == list: Ports = Ports.copy()
		elif type(Ports) == int: Ports = [Ports]

		Portstr = ""
		for Port in Ports:
			if Port in self.Ports:
				Portstr += "{} ".format(Port)
				self.Ports.remove(Port)

		Salome_close = Popen('{} kill {}'.format(self.Exec,Portstr), shell = 'TRUE')
		self.Logger('Closing Salome on port(s) {}'.format(Ports))

		return Salome_close

	def Kill(self, Port=0, PortFile=''):
		if Port:
			KillPort=Port
		elif PortFile:
			with open(PortFile,'r') as f:
				KillPort = int(f.readline())
		else : KillPort = 0


		if True:
			cmlst = self.Exec.split() + ['kill', str(KillPort)]
			SubProc = Popen(cmlst)
		else :
			SubProc = Popen("{} kill {}".format(self.Exec, KillPort), shell='TRUE')
		SubProc.wait()

	def Run(self, Script, **kwargs):
		# AddPath will always add these paths to salome environment
		# self.AddPath = kwargs.get('AddPath',[]) + ["{}/VLPackages/Salome".format(self.COM_SCRIPTS)]

		'''
		kwargs available:
		OutFile: The log file you want to write stdout to (default is /dev/null)
		AddPath: Additional paths that Salome will be able to import from
		ArgDict: a dictionary of the arguments that Salome will get
		ArgList: a list of arguments to be passed to Salome
		GUI: Opens a new instance with GUI (useful for testing)
		'''
		AddPath = kwargs.get('AddPath',[])
		ArgDict = kwargs.get('ArgDict', {})
		ArgList = kwargs.get('ArgList',[])
		OutFile = kwargs.get('OutFile', None)

		# Add paths provided to python path for subprocess (self.COM_SCRIPTS and self.SIM_SCRIPTS is always added to path)
		AddPath = [AddPath] if type(AddPath) == str else AddPath
		PyPath = ["{}:".format(path) for path in AddPath+self.AddPath]
		PyPath = "".join(PyPath)

		# Write ArgDict and ArgList in format to pass to salome
		Args = ["{}={}".format(key, value) for key, value in ArgDict.items()]
		Args = ",".join(ArgList + Args)

		portfile = "{}/{}".format(self.cwd,uuid.uuid4())
		GUIflag = '-g' if kwargs.get('GUI') else '-t'

		env = {**os.environ, 'PYTHONPATH': PyPath + os.environ['PYTHONPATH']}
		# Run mesh in Salome
		if False:
			# This is dev work, need to add in output option for this call
			cmlst = self.Exec.split() + [GUIflag, '--ns-port-log', portfile, Script, 'args:'+Args]
			SubProc = Popen(cmlst, cwd=self.cwd, stdout=sys.stdout, stderr=sys.stderr, env=env)
		else :
			output = ">>{} 2>&1".format(OutFile) if OutFile else ''
			command = "{} {} --ns-port-log {} {} args:{} {}".format(self.Exec, GUIflag, portfile, Script, Args, output)
			SubProc = Popen(command, shell='TRUE',cwd=self.cwd,env=env)
		ReturnCode = SubProc.wait()

		self.Kill(PortFile=portfile)

		return ReturnCode
