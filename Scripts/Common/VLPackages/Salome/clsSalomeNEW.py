#!/usr/bin/env python3
import tempfile

class Salome():
	def __init__(self, super):
		self.TMP_DIR = super.TMP_DIR
		self.COM_SCRIPTS = super.COM_SCRIPTS
		self.SIM_SCRIPTS = super.SIM_SCRIPTS
		self.Logger = super.Logger
		self.Exit = super.Exit
		self.Ports = []
		self.LogFile = super.LogFile

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
		OutFile = ErrFile = kwargs.get('OutFile', self.LogFile)
		ErrFile = kwargs.get('ErrFile',ErrFile)

		output = ''
		if OutFile: output += " >>{}".format(OutFile)
		if ErrFile: output += " 2>>{}".format(ErrFile)

		self.Logger("Initiating Salome\n", Print=True)

		SalomeSP = []
		NewPorts = []
		for i in range(Num):
			portfile = "{}/{}".format(self.TMP_DIR,uuid.uuid4())
			SubProc = Popen('cd {};salome -t --ns-port-log {} {}'.format(self.TMP_DIR, portfile, output), shell='TRUE')
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

	def Run(self, Script, **kwargs):
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
		AddPath += [self.COM_SCRIPTS, self.SIM_SCRIPTS]
		PythonPath = ["PYTHONPATH={}:$PYTHONPATH;".format(path) for path in AddPath]
		PythonPath.append("export PYTHONPATH;")
		PythonPath = "".join(PythonPath)

		# Write ArgDict and ArgList in format to pass to salome
		Args = ["{}={}".format(key, value) for key, value in ArgDict.items()]
		Args = ",".join(ArgList + Args)

		if kwargs.get('GUI',False):
			command = "salome {} args:{}".format(Script, Args)
			SubProc = Popen(PythonPath + command, shell='TRUE')
			return SubProc



		OutFile = ErrFile = kwargs.get('OutFile', self.LogFile)
		ErrFile = kwargs.get('ErrFile',ErrFile)

		output = ''
		if OutFile: output += " >>{}".format(OutFile)
		if ErrFile: output += " 2>>{}".format(ErrFile)

		salomedir = tempfile.mkdtemp(suffix = 'Salome', dir=self.TMP_DIR)
		command = "cd {0};salome -t --ns-port-log {0}/portfile.txt {}".format(salomedir)

		SubProc = Popen('cd {};salome -t --ns-port-log {} {}'.format(self.TMP_DIR, portfile, output), shell='TRUE')




		# if kwargs.get('Shell',False):
		# 	if not self.Ports:
		# 		self.Start()
		# 	Port = kwargs.get('Port', self.Ports[0])
		# 	command = "salome shell -p{!s} {} args:{} {}".format(Port, Script, Args, output)
		# else :
		# 	command = "salome {} args:{} {}".format(Script, Args, output)

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

		Salome_close = Popen('salome kill {}'.format(Portstr), shell = 'TRUE')
		self.Logger('Closing Salome on port(s) {}'.format(Ports))

		return Salome_close