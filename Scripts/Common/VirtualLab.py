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
from Scripts.Common import Analytics
from Scripts.Common.VLPackages import Salome, CodeAster
from Scripts.Common.VLTypes import Mesh as MeshFn, Sim as SimFn, DA as DAFn

class VLSetup():
	def __init__(self, Simulation, Project, StudyName, Parameters_Master=None, Parameters_Var=None, Mode='T',
				 InputDir=None, OutputDir=None, MaterialDir=None, TempDir=None):

		self.Simulation = Simulation
		self.Project = Project
		self.StudyName = StudyName
		# Parameters_Master and Var will be overwritten by a namespace of the parameters using function CreateParameters
		self._Parameters_Master = Parameters_Master
		self._Parameters_Var = Parameters_Var
		self.mode = Mode

		### Define required directories for VL from config file. If directory name doesn't start with '/'
		### it will be created relative to the TWD.
		# Output directory - this is where meshes, Aster results and pre/post-processing will be stored.
		# Material directory
		# Input directory
		# Temp directory
		VL_DIR = VLconfig.VL_DIR
		_InputDir = getattr(VLconfig,'InputDir', "{}/Input".format(VL_DIR))
		_OutputDir = getattr(VLconfig,'OutputDir', "{}/Output".format(VL_DIR))
		_MaterialDir = getattr(VLconfig,'MaterialDir', "{}/Materials".format(VL_DIR))
		_TempDir = getattr(VLconfig,'TEMP_DIR',"/tmp")

		# Set directories to those set in kwargs if given else
		self.INPUT_DIR=InputDir if InputDir else _InputDir
		self.OUTPUT_DIR=OutputDir if OutputDir else _OutputDir
		self.MATERIAL_DIR=MaterialDir if MaterialDir else _MaterialDir
		self.TEMP_DIR=TempDir if TempDir else _TempDir

		# Update above with parsed arguments
		for key, val in self.GetArgParser().items():
			if key in self.__dict__:
				setattr(self,key,val)
			if key == 'Mode':
				self.mode = val

		# Update running mode as shorthand version can be given
		if self.mode.lower() in ('i', 'interactive'): self.mode = 'Interactive'
		elif self.mode.lower() in ('t','terminal'): self.mode = 'Terminal'
		elif self.mode.lower() in ('c', 'continuous'): self.mode = 'Continuous'
		elif self.mode.lower() in ('h', 'headless'): self.mode = 'Headless'
		else : self.Exit("Error: Mode is not in; 'Interactive','Terminal','Continuous' or 'Headless'")

		# Remove RunFiles directory from path if a scrit was launched there.
		# if VL_DIR != sys.path[-1]: sys.path.pop(-1)

		self.__ID__ = (datetime.datetime.now()).strftime("%y.%m.%d_%H.%M.%S.%f")

		# Update Input, Output and Temp directories with simulation specific ones
		self.TEMP_DIR = '{}/VL_{}'.format(self.TEMP_DIR, self.__ID__)
		try:
			os.makedirs(self.TEMP_DIR)
		except FileExistsError:
			pass

		self.INPUT_DIR = '{}/{}/{}'.format(self.INPUT_DIR, Simulation, Project)

		# Output directory
		self.PROJECT_DIR = "{}/{}/{}".format(self.OUTPUT_DIR, Simulation, Project)
		self.STUDY_DIR = "{}/{}".format(self.PROJECT_DIR, StudyName)
		self.MESH_DIR = "{}/Meshes".format(self.PROJECT_DIR)


		# Define variables and run some checks
		# Script directories
		self.COM_SCRIPTS = "{}/Scripts/Common".format(VL_DIR)
		self.SIM_SCRIPTS = "{}/Scripts/{}".format(VL_DIR, Simulation)

		# Scrpt directories
		self.SIM_MESH = "{}/Mesh".format(self.SIM_SCRIPTS)
		self.SIM_PREASTER = "{}/PreAster".format(self.SIM_SCRIPTS)
		self.SIM_ASTER = "{}/Aster".format(self.SIM_SCRIPTS)
		self.SIM_POSTASTER = "{}/PostAster".format(self.SIM_SCRIPTS)
		self.SIM_DA = "{}/DA".format(self.SIM_SCRIPTS)

		self._pypath = sys.path.copy() # Needed for MPI run to match sys.path

		self.Logger('### Launching VirtualLab ###',Print=True)

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

		if not hasattr(self,'Salome'):
			self.Salome = Salome.Salome(self, AddPath=[self.SIM_SCRIPTS])
		if not hasattr(self,'CodeAster'):
			self.CodeAster = CodeAster.CodeAster(self)

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
		return MeshFn.devRun(self,**kwargs)

	def Sim(self,**kwargs):
		return SimFn.Run(self,**kwargs)

	def devSim(self,**kwargs):
		return SimFn.devRun(self,**kwargs)

	def devDA(self,**kwargs):
		return DAFn.devRun(self,**kwargs)

	def WriteModule(self, FileName, Dictionary, **kwargs):
		Write = kwargs.get('Write','New')
		if Write == 'New':
			PathList = []
			for VarName, Val in Dictionary.items():
				if type(Val)==str: Val = "'{}'".format(Val)
				PathList.append("{} = {}\n".format(VarName, Val))
			Pathstr = ''.join(PathList)
			with open(FileName,'w+') as f:
				f.write(Pathstr)

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
		self.Logger(mess, Print=True)
		self.Cleanup(KeepDirs)
		sys.exit()

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

		Abs_Parameters = "{}/{}.py".format(self.INPUT_DIR,Rel_Parameters)
		# Check File exists
		if not os.path.exists(Abs_Parameters):
			message = "The following Parameter file does not exist:\n{}".format(Abs_Parameters)
			self.Exit(self._Error(message))

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
			self.Exit(self._Error(message))

		Var=getattr(Parameters_Var,InstName, None)
		if not Var: return {Master.Name : Master.__dict__}
		if not hasattr(Var,'Name'):
			message = "'{}' does not have the attribute 'Name' in Parameters_Var".format(InstName)
			self.Exit(self._Error(message))

		# Check if there are attributes defined in Var which are not in Master
		dfattrs = set(Var.__dict__.keys()) - set(list(Master.__dict__.keys())+['Run'])
		if dfattrs:
			attstr = "\n".join(["{}.{}".format(InstName,i) for i in dfattrs])
			message = "The following attribute(s) are specified in Parameters_Var but not in Parameters_Master:\n"\
				"{}\n\nThis may lead to unexpected results.".format(attstr)
			print(self._Warning(message))

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
			self.Exit(self._Error(message))

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

	def _Warning(self, message):
		warning = "\n======== Warning ========\n\n"\
			"{}\n\n"\
			"=========================\n\n".format(message)
		return warning

	def _Error(self,message):
		error = "\n========= Error =========\n\n"\
			"{}\n\n"\
			"=========================\n\n".format(message)
		return error
