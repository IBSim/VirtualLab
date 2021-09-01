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
	def __init__(self, Simulation, Project, StudyName, Parameters_Master=None, Parameters_Var=None,
				 Mode='T', InputDir=VLconfig.InputDir, OutputDir=VLconfig.OutputDir,
				 MaterialDir=VLconfig.MaterialsDir, TempDir=VLconfig.TEMP_DIR):

		# Parameters can be overwritten using parsed arguments
		ParsedArgs = self.GetArgParser()

		self.Simulation = ParsedArgs.get('Simulation',Simulation)
		self.Project = ParsedArgs.get('Project',Project)
		self.StudyName = ParsedArgs.get('StudyName',StudyName)

		self.Paths(VLconfig.VL_DIR,
				   ParsedArgs.get('InputDir',InputDir),
				   ParsedArgs.get('OutputDir',OutputDir),
				   ParsedArgs.get('MaterialDir',MaterialDir),
				   ParsedArgs.get('TempDir',TempDir))

		# This is to ensure the Control method works
		self._Parameters_Master = ParsedArgs.get('Parameters_Master',Parameters_Master)
		self._Parameters_Var = ParsedArgs.get('Parameters_Var',Parameters_Var)

		self.mode = ParsedArgs.get('Mode',Mode)
		# Update mode as shorthand version can be given
		if self.mode.lower() in ('i', 'interactive'): self.mode = 'Interactive'
		elif self.mode.lower() in ('t','terminal'): self.mode = 'Terminal'
		elif self.mode.lower() in ('c', 'continuous'): self.mode = 'Continuous'
		elif self.mode.lower() in ('h', 'headless'): self.mode = 'Headless'
		else : self.Exit("Error: Mode is not in; 'Interactive','Terminal','Continuous' or 'Headless'")

		self._pypath = sys.path.copy() # Needed for MPI run to match sys.path

		self.Logger('### Launching VirtualLab ###',Print=True)

	def Paths(self,VLDir,InputDir,OutputDir,MaterialDir,TempDir):
		'''
		Paths to important locations within VirtualLab are defined in this function.
		'''

		# Define path to Parameters file
		self._InputDir = InputDir
		self.PARAMETERS_DIR = '{}/{}/{}'.format(self._InputDir, self.Simulation, self.Project)

		# Define paths to results
		self._OutputDir = OutputDir
		self.PROJECT_DIR = "{}/{}/{}".format(self._OutputDir, self.Simulation, self.Project)
		self.STUDY_DIR = "{}/{}".format(self.PROJECT_DIR, self.StudyName)

		# Define path to Materials
		self.MATERIAL_DIR = MaterialDir

		# Define & create temporary directory for work to be saved to
		self._TempDir = TempDir
		# Unique ID
		self.__ID__ = (datetime.datetime.now()).strftime("%y.%m.%d_%H.%M.%S.%f")

		self.TEMP_DIR = '{}/VL_{}'.format(self._TempDir, self.__ID__)
		try:
			os.makedirs(self.TEMP_DIR)
		except FileExistsError:
			# Unlikely this would happen. Suffix random number to direcory name
			self.TEMP_DIR = "{}_{}".format(self.TEMP_DIR,np.random.random_integer(1000))
			os.makedirs(self.TEMP_DIR)

		# Define paths to script directories
		self.COM_SCRIPTS = "{}/Scripts/Common".format(VLDir)
		self.SIM_SCRIPTS = "{}/Scripts/{}".format(VLDir, self.Simulation)

	def Parameters(self, Parameters_Master, Parameters_Var=None,
					RunMesh=True, RunSim=True, RunDA=True):
		'''
		This method is replacing control.
		'''

		kw = {'RunMesh':RunMesh,'RunSim':RunSim,'RunDA':RunDA}
		kw.update(self.GetArgParser())

		# Create variables based on the namespaces (NS) in the Parameters file(s) provided
		VLNamespaces = ['Mesh','Sim','DA']
		self.GetParams(Parameters_Master, Parameters_Var, VLNamespaces)

		sys.path = [self.COM_SCRIPTS,self.SIM_SCRIPTS] + sys.path

		MeshFn.Setup(self,**kw)
		SimFn.Setup(self,**kw)
		DAFn.Setup(self,**kw)

		# Function to analyse usage of VirtualLab to evidence impact for
		# use in future research grant applications. Can be turned off via
		# VLconfig.py. See Scripts/Common/Analytics.py for more details.
		if VLconfig.VL_ANALYTICS=="True": Analytics.Run(self,**kw)

	def Control(self, **kwargs):
		'''
		kwargs available:
		RunMesh: Boolean to dictate whether or not to create meshes
		RunSim: Boolean to dictate whether or not to run simulation routine
		RunDA: Boolean to dictate data analysis part (dev)

		This method will be depreciated in future.
		'''
		if self._Parameters_Master == None:
			self.Exit('Parameters_Master/var must be set during class initiation to use this method')

		self.Parameters(self._Parameters_Master,self._Parameters_Var,**kwargs)


	def Mesh(self,**kwargs):
		return MeshFn.Run(self,**kwargs)

	def devMesh(self,**kwargs):
		return MeshFn.Run(self,**kwargs)

	def Sim(self,**kwargs):
		return SimFn.Run(self,**kwargs)

	def devSim(self,**kwargs):
		return SimFn.Run(self,**kwargs)

	def DA(self,**kwargs):
		return DAFn.Run(self,**kwargs)

	def devDA(self,**kwargs):
		return DAFn.Run(self,**kwargs)

	def Logger(self,Text='',**kwargs):
		Prnt = kwargs.get('Print',False)

		if not hasattr(self,'LogFile'):
			print(Text)
			if self.mode in ('Interactive','Terminal'):
				self.LogFile = None
			else:
				self.LogFile = "{}/.log/{}_{}.log".format(self.PROJECT_DIR,self.StudyName,self.__ID__)
				os.makedirs(os.path.dirname(self.LogFile), exist_ok=True)
				with open(self.LogFile,'w') as f:
					f.write(Text)
				print("Detailed output written to {}".format(self.LogFile))
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

		if hasattr(self, 'Salome'):
			if self.Salome.Ports:
				self.Salome.Close(self.Salome.Ports)

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

	def GetArgParser(self):
		ArgList=sys.argv[1:]
		argdict={}
		for arg in ArgList:
			split=arg.split('=')
			if len(split)!=2:
				continue
			var,value = split
			if value=='False':value=False
			elif value=='True':value=True
			elif value=='None':value=None
			elif value.isnumeric():value=int(value)
			else:
				try:
					value=float(value)
				except: ValueError

			argdict[var]=value

		return argdict
