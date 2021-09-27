#!/usr/bin/env python3

import sys
sys.dont_write_bytecode=True
import datetime
import os
import numpy as np
import shutil
import copy
from types import SimpleNamespace as Namespace
from importlib import import_module, reload

import VLconfig
from . import Analytics
from .VLFunctions import ErrorMessage, WarningMessage
from .VLTypes import Mesh as MeshFn, Sim as SimFn, DA as DAFn

class VLSetup():
	def __init__(self, Simulation, Project):

		# Get parsed args (achieved using the -k flag when launchign VirtualLab).
		self._GetParsedArgs()
		# Copy path at the start for MPI to match sys.path
		self._pypath = sys.path.copy()

		# ======================================================================
		# Define variables
		self.Simulation = self._ParsedArgs.get('Simulation',Simulation)
		self.Project = self._ParsedArgs.get('Project',Project)

		# ======================================================================
		# Specify default settings
		self.Settings(Mode='H',Launcher='Process',NbThreads=1,
					  InputDir=VLconfig.InputDir, OutputDir=VLconfig.OutputDir,
					  MaterialDir=VLconfig.MaterialsDir)

		# ======================================================================
		# Define path to scripts
		self.COM_SCRIPTS = "{}/Scripts/Common".format(VLconfig.VL_DIR)
		self.SIM_SCRIPTS = "{}/Scripts/{}".format(VLconfig.VL_DIR, self.Simulation)
		# Add these to path
		sys.path = [self.COM_SCRIPTS,self.SIM_SCRIPTS] + sys.path

		#=======================================================================
		# Define & create temporary directory for work to be saved to
		self._TempDir = VLconfig.TEMP_DIR
		# Unique ID
		self.__ID__ = (datetime.datetime.now()).strftime("%y.%m.%d_%H.%M.%S.%f")

		self.TEMP_DIR = '{}/VL_{}'.format(self._TempDir, self.__ID__)
		try:
			os.makedirs(self.TEMP_DIR)
		except FileExistsError:
			# Unlikely this would happen. Suffix random number to direcory name
			self.TEMP_DIR = "{}_{}".format(self.TEMP_DIR,np.random.random_integer(1000))
			os.makedirs(self.TEMP_DIR)

		self.Logger('### Launching VirtualLab ###',Print=True)


	def _SetMode(self,Mode='H'):
		Mode = self._ParsedArgs.get('Mode',Mode)
		# ======================================================================
		# Update mode as shorthand version can be given
		if Mode.lower() in ('i', 'interactive'): self.mode = 'Interactive'
		elif Mode.lower() in ('t','terminal'): self.mode = 'Terminal'
		elif Mode.lower() in ('c', 'continuous'): self.mode = 'Continuous'
		elif Mode.lower() in ('h', 'headless'): self.mode = 'Headless'
		else : self.Exit(ErrorMessage("Mode must be one of; 'Interactive',\
									  'Terminal','Continuous', 'Headless'"))

	def _SetLauncher(self,Launcher='Process'):
		Launcher = self._ParsedArgs.get('Launcher',Launcher)
		if Launcher.lower() == 'sequential': self._Launcher = 'Sequential'
		elif Launcher.lower() == 'process': self._Launcher = 'Process'
		elif Launcher.lower() == 'mpi': self._Launcher = 'MPI'
		else: self.Exit(ErrorMessage("Launcher must be one of; 'Sequential',\
									 'Process', 'MPI'"))

	def _SetNbThreads(self,NbThreads=1):
		NbThreads = self._ParsedArgs.get('NbThreads',NbThreads)
		if type(NbThreads) == int:
			_NbThreads = NbThreads
		elif type(NbThreads) == float:
			if NbThreads.is_integer():
				_NbThreads = NbThreads
			else:
				self.Exit(ErrorMessage("NbThreads must be an integer"))
		else:
			self.Exit(ErrorMessage("NbThreads must be an integer"))

		if _NbThreads >= 1:
			self._NbThreads = _NbThreads
		else:
			self.Exit(ErrorMessage("NbThreads must be positive"))

	def _SetInputDir(self,InputDir):
		InputDir = self._ParsedArgs.get('InputDir',InputDir)
		if not os.path.isdir(InputDir):
			self.Exit(ErrorMessage("InputDir is not a valid directory"))
		self._InputDir = InputDir
		self.PARAMETERS_DIR = '{}/{}/{}'.format(self._InputDir, self.Simulation, self.Project)

	def _SetOutputDir(self,OutputDir):
		OutputDir = self._ParsedArgs.get('OutputDir',OutputDir)
		self._OutputDir = OutputDir
		self.PROJECT_DIR = '{}/{}/{}'.format(self._OutputDir, self.Simulation, self.Project)

	def _SetMaterialDir(self,MaterialDir):
		if not os.path.isdir(MaterialDir):
			self.Exit(ErrorMessage("MaterialDir is not a valid directory"))
		MaterialDir = self._ParsedArgs.get('MaterialDir',MaterialDir)
		self.MATERIAL_DIR = MaterialDir

	def Settings(self,**kwargs):

		Diff = set(kwargs).difference(['Mode','Launcher','NbThreads','InputDir',
									'OutputDir','MaterialDir'])
		if Diff:
			self.Exit("Error: {} are not option(s) for settings".format(list(Diff)))

		if 'Mode' in kwargs:
			self._SetMode(kwargs['Mode'])
		if 'Launcher' in kwargs:
			self._SetLauncher(kwargs['Launcher'])
		if 'NbThreads' in kwargs:
			self._SetNbThreads(kwargs['NbThreads'])
		if 'InputDir' in kwargs:
			self._SetInputDir(kwargs['InputDir'])
		if 'OutputDir' in kwargs:
			self._SetOutputDir(kwargs['OutputDir'])
		if 'MaterialDir' in kwargs:
			self._SetMaterialDir(kwargs['MaterialDir'])

	def Parameters(self, Parameters_Master, Parameters_Var=None,
					RunMesh=True, RunSim=True, RunDA=True):

		# Update args with parsed args
		Parameters_Master = self._ParsedArgs.get('Parameters_Master',Parameters_Master)
		Parameters_Var = self._ParsedArgs.get('Parameters_Var',Parameters_Var)
		RunMesh = self._ParsedArgs.get('RunMesh',RunMesh)
		RunSim = self._ParsedArgs.get('RunSim',RunSim)
		RunDA = self._ParsedArgs.get('RunDA',RunDA)

		# Create variables based on the namespaces (NS) in the Parameters file(s) provided
		VLNamespaces = ['Mesh','Sim','DA']
		self.GetParams(Parameters_Master, Parameters_Var, VLNamespaces)

		MeshFn.Setup(self,RunMesh)
		SimFn.Setup(self,RunSim)
		DAFn.Setup(self,RunDA)

	def Mesh(self,**kwargs):
		kwargs = self._UpdateArgs(kwargs)
		return MeshFn.Run(self,**kwargs)

	def devMesh(self,**kwargs):
		kwargs = self._UpdateArgs(kwargs)
		return MeshFn.Run(self,**kwargs)

	def Sim(self,**kwargs):
		kwargs = self._UpdateArgs(kwargs)
		return SimFn.Run(self,**kwargs)

	def devSim(self,**kwargs):
		kwargs = self._UpdateArgs(kwargs)
		return SimFn.Run(self,**kwargs)

	def DA(self,**kwargs):
		kwargs = self._UpdateArgs(kwargs)
		return DAFn.Run(self,**kwargs)

	def devDA(self,**kwargs):
		kwargs = self._UpdateArgs(kwargs)
		return DAFn.Run(self,**kwargs)

	def Logger(self,Text='',**kwargs):
		Prnt = kwargs.get('Print',False)

		if not hasattr(self,'LogFile'):
			print(Text)
			if self.mode in ('Interactive','Terminal'):
				self.LogFile = None
			else:
				self.LogFile = "{}/.log/{}.log".format(self.PROJECT_DIR, self.__ID__)
				os.makedirs(os.path.dirname(self.LogFile), exist_ok=True)
				with open(self.LogFile,'w') as f:
					f.write(Text)
				# print("Detailed output written to {}".format(self.LogFile))
			return

		if self.mode in ('Interactive','Terminal'):
			print(Text,flush=True)
		else:
			if Prnt: print(Text,flush=True)
			with open(self.LogFile,'a') as f:
				f.write(Text+"\n")

	def Exit(self,mess='',KeepDirs=[]):
		# self.Logger(mess, Print=True)
		self.Cleanup(KeepDirs)
		sys.exit(mess)

	def Cleanup(self,KeepDirs=[]):
		# Report overview of VirtualLab usage
		if hasattr(self,'_Analytics') and VLconfig.VL_ANALYTICS=="True":
			MeshNb = self._Analytics.get('Mesh',0)
			SimNb = self._Analytics.get('Sim',0)
			DANb = self._Analytics.get('DANb',0)
			Category = "{}_Overview".format(self.Simulation)
			Action = "{}_{}_{}".format(MeshNb,SimNb,DANb)
			Label = self.__ID__
			Analytics.Run(Category,Action,Label)

		if os.path.isdir(self.TEMP_DIR):
			if KeepDirs:
				kept = []
				for ct in os.listdir(self.TEMP_DIR):
					SubDir = '{}/{}'.format(self.TEMP_DIR,ct)
					if os.path.isdir(SubDir):
						if ct in KeepDirs: kept.append(SubDir)
						else : shutil.rmtree(SubDir)
				self.Logger("The following tmp directories have not been deleted:\n{}".format(kept),Print=True)
			else:
				shutil.rmtree(self.TEMP_DIR)

		# self.Logger('### VirtualLab Finished###\n',Print=True)

	def ImportParameters(self, Rel_Parameters):
		'''
		Rel_Parameters is a file name relative to the Input directory
		'''
		# Strip .py off the end if it's in the name
		if os.path.splitext(Rel_Parameters)[1]=='.py':
			Rel_Parameters = os.path.splitext(Rel_Parameters)[0]

		Abs_Parameters = "{}/{}.py".format(self.PARAMETERS_DIR,Rel_Parameters)
		# Check File exists
		if not os.path.exists(Abs_Parameters):
			message = "The following Parameter file does not exist:\n{}".format(Abs_Parameters)
			self.Exit(ErrorMessage(message))

		sys.path.insert(0, os.path.dirname(Abs_Parameters))
		Parameters = reload(import_module(os.path.basename(Rel_Parameters)))
		sys.path.pop(0)

		return Parameters

	def GetParams(self, Parameters_Master, Parameters_Var, NS):
		# Parameters_Master &/or Var can be module, a namespace or string.
		# A string references a file in the input directory

		# ======================================================================
		# Parameters Master
		if type(Parameters_Master)==str:
			Main = self.ImportParameters(Parameters_Master)
		elif any(hasattr(Parameters_Master,nm) for nm in NS):
			Main = Parameters_Master
		else: sys.exit()

		self.Parameters_Master = Namespace()
		for nm in NS:
			setattr(self.Parameters_Master, nm, getattr(Main, nm, None))

		# ======================================================================
		# Parameters Var
		if type(Parameters_Var)==str:
			Var = self.ImportParameters(Parameters_Var)
		elif any(hasattr(Parameters_Var,nm) for nm in NS):
			Var = Parameters_Var
		elif Parameters_Var==None:
			Var = None
		else: sys.exit()

		self.Parameters_Var = Namespace()
		for nm in NS:
			setattr(self.Parameters_Var, nm, getattr(Var, nm, None))

		# ======================================================================

	def CreateParameters(self, Parameters_Master, Parameters_Var, InstName):
		'''
		Create parameter dictionary for instance 'InstName' using Parameters_Master and Var.
		'''
		# ======================================================================
		# Performs Checks & return warnings or errors

		# Get instance 'InstName' from Parameters_Master and _Parameters_Var (if they are defined)
		# and check they have the attribute 'Name'
		Master=getattr(Parameters_Master,InstName, None)
		if not Master: return {}
		if not hasattr(Master,'Name'):
			message = "'{}' does not have the attribute 'Name' in Parameters_Master".format(InstName)
			self.Exit(ErrorMessage(message))

		Var=getattr(Parameters_Var,InstName, None)
		if not Var: return {Master.Name : Master.__dict__}
		if not hasattr(Var,'Name'):
			message = "'{}' does not have the attribute 'Name' in Parameters_Var".format(InstName)
			self.Exit(ErrorMessage(message))

		# Check if there are attributes defined in Var which are not in Master
		dfattrs = set(Var.__dict__.keys()) - set(list(Master.__dict__.keys())+['Run'])
		if dfattrs:
			attstr = "\n".join(["{}.{}".format(InstName,i) for i in dfattrs])
			message = "The following attribute(s) are specified in Parameters_Var but not in Parameters_Master:\n"\
				"{}\n\nThis may lead to unexpected results.".format(attstr)
			print(WarningMessage(message))

		# Check all entires in Parameters_Var have the same length
		NbNames = len(Var.Name)
		VarNames, NewVals, errVar = [],[], []
		for VariableName, NewValues in Var.__dict__.items():
			VarNames.append(VariableName)
			NewVals.append(NewValues)
			if len(NewValues) != NbNames:
				errVar.append(VariableName)

		if errVar:
			attrstr = "\n".join(["{}.{}".format(InstName,i) for i in errVar])
			message = "The following attribute(s) have a different number of entries to {0}.Name in Parameters_Var:\n"\
				"{1}\n\nAll attributes of {0} in Parameters_Var must have the same length.".format(InstName,attrstr)
			self.Exit(ErrorMesage(message))

		# ======================================================================
		# Create dictionary for each entry in Parameters_Var
		VarRun = getattr(Var,'Run',[True]*NbNames) # create True list if Run not an attribute of InstName
		ParaDict = {}
		for Name, NewValues, Run in zip(Var.Name,zip(*NewVals),VarRun):
			if not Run: continue

			cpMaster = copy.deepcopy(Master.__dict__)
			for VariableName, NewValue in zip(VarNames,NewValues):
				# if type(NewValue)==dict:
				cpMaster[VariableName]=NewValue
			ParaDict[Name] = cpMaster

		return ParaDict

	def _GetParsedArgs(self):
		self._ParsedArgs = {}
		for arg in sys.argv[1:]:
			split=arg.split('=')
			if len(split)!=2:
				continue
			var,value = split
			if value=='False':value=False
			elif value=='True':value=True
			elif value=='None':value=None
			elif value.isnumeric():value=int(value)
			else:
				try: value=float(value)
				except: ValueError

			self._ParsedArgs[var]=value

	def _UpdateArgs(self,ArgDict):
		Changes = set(ArgDict).intersection(self._ParsedArgs)
		if not Changes: return ArgDict

		# If some of the arguments have been parsed then they are updated
		for key in Changes:
			ArgDict[key] = self._ParsedArgs[key]
		return ArgDict
