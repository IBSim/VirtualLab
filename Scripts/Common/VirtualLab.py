#!/usr/bin/env python3

import sys
sys.dont_write_bytecode=True
import datetime
import os
import numpy as np
import shutil
import copy
from types import SimpleNamespace as Namespace
from importlib import import_module

import VLconfig
from Scripts.Common import Analytics
from Scripts.Common.VLPackages import Salome, CodeAster
from Scripts.Common.VLTypes import Mesh as MeshFn, Sim as SimFn, ML as MLFn

class VLSetup():
	def __init__(self, Simulation, Project, StudyName, Parameters_Master, Parameters_Var=None, Mode='T'):

		self.Simulation = Simulation
		self.Project = Project
		self.StudyName = StudyName
		# Parameters_Master and Var will be overwritten by a namespace of the parameters using function CreateParameters
		self.Parameters_Master = Parameters_Master
		self.Parameters_Var = Parameters_Var
		self.mode = Mode

		VL_DIR = VLconfig.VL_DIR

		### Define directories for VL from config file. If directory name doesn't start with '/'
		### it will be created relative to the TWD
		# Output directory - this is where meshes, Aster results and pre/post-processing will be stored.
		self.OUTPUT_DIR = getattr(VLconfig,'OutputDir', "{}/Output".format(VL_DIR))
		# Material directory
		self.MATERIAL_DIR = getattr(VLconfig,'MaterialDir', "{}/Materials".format(VL_DIR))
		# Input directory
		self.INPUT_DIR = getattr(VLconfig,'InputDir', "{}/Input".format(VL_DIR))
		# tmp directory
		self.TEMP_DIR = getattr(VLconfig,'TEMP_DIR',"/tmp")

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
		if VL_DIR != sys.path[-1]: sys.path.pop(-1)

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
		self.SIM_ML = "{}/ML".format(self.SIM_SCRIPTS)

		self.Logger('### Launching VirtualLab ###',Print=True)

	def Control(self, **kwargs):
		'''
		kwargs available:
		RunMesh: Boolean to dictate whether or not to create meshes
		RunSim: Boolean to dictate whether or not to run simulation routine
		RunMl: Boolean to dictate ML part (dev)
		'''
		kwargs.update(self.GetArgParser())

		# Create variables based on the namespaces (NS) in the Parameters file(s) provided
		self.NS = ['Mesh','Sim','ML']
		self.GetParams(self.Parameters_Master, self.Parameters_Var, self.NS)

		self.Salome = Salome.Salome(self, AddPath=[self.SIM_SCRIPTS])
		self.CodeAster = CodeAster.CodeAster(self)

		sys.path = [self.COM_SCRIPTS,self.SIM_SCRIPTS] + sys.path

		MeshFn.Setup(self,**kwargs)
		SimFn.Setup(self,**kwargs)
		MLFn.Setup(self,**kwargs)

		# Function to analyse usage of VirtualLab to evidence impact for
		# use in future research grant applications. Can be turned off via
		# VLconfig.py. See Scripts/Common/Analytics.py for more details.
		if VLconfig.VL_ANALYTICS=="True": Analytics.Run(self,**kwargs)


	def Mesh(self,**kwargs):
		return MeshFn.Run(self,**kwargs)

	def devMesh(self,**kwargs):
		return MeshFn.devRun(self,**kwargs)

	def Sim(self,**kwargs):
		return SimFn.Run(self,**kwargs)

	def devSim(self,**kwargs):
		return SimFn.devRun(self,**kwargs)

	def devML(self,**kwargs):
		return MLFn.devRun(self,**kwargs)



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
			print(Text)
		else:
			if Prnt: print(Text)
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
	def GetParams(self, Parameters_Master, Parameters_Var, NS):
		sys.path.insert(0, self.INPUT_DIR)

		if type(Parameters_Master)==str:
			if not os.path.exists('{}/{}.py'.format(self.INPUT_DIR, Parameters_Master)):
				self.Exit("Parameters_Master file '{}' not in directory {}".format(Parameters_Master, self.INPUT_DIR))
			Main = import_module(Parameters_Master)
		elif any(hasattr(Parameters_Master,nm) for nm in NS):
			Main = Parameters_Master
		else: sys.exit()
		self.Parameters_Master = Namespace()
		for nm in NS:
			setattr(self.Parameters_Master, nm, getattr(Main, nm, None))

		if type(Parameters_Var)==str:
			if not os.path.exists('{}/{}.py'.format(self.INPUT_DIR, Parameters_Var)):
				self.Exit("Parameters_Var file '{}' not in directory {}".format(Parameters_Var, self.INPUT_DIR))
			Var = import_module(Parameters_Var)
		elif any(hasattr(Parameters_Var,nm) for nm in NS):
			Var = Parameters_Var
		elif Parameters_Var==None:
			Var = None
		else: sys.exit()
		self.Parameters_Var = Namespace()
		for nm in NS:
			setattr(self.Parameters_Var, nm, getattr(Var, nm, None))

	def CreateParameters(self, Parameters_Master, Parameters_Var, Attr):
		Master=getattr(Parameters_Master,Attr, None)
		if not Master: return {}
		if not hasattr(Master,'Name'): self.Exit("Error: '{}' does not have the attribute 'Name' in Parameters_Master".format(Attr))

		Var=getattr(Parameters_Var,Attr, None)
		if not Var: return {Master.Name : Master.__dict__}
		if not hasattr(Var,'Name'): self.Exit("Error: '{}' does not have the attribute 'Name' in Parameters_Var".format(Attr))

		NbNames = len(Var.Name)
		ParaDict = {}
		for VariableName, MasterValue in Master.__dict__.items():
			# check types
			NewValues = getattr(Var, VariableName, [MasterValue]*NbNames)
			# Check the number of NewVals is correct
			if len(NewValues) != NbNames:
				self.Exit("Error: Number of entries for '{0}.{1}' not equal to '{0}.Names' in Parameters_Var".format(Attr,VariableName))
			for Name,NewVal in zip(Var.Name,NewValues):
				if type(MasterValue)==dict:
					cpdict = copy.deepcopy(MasterValue)
					cpdict.update(NewVal)
					NewVal=cpdict
					DiffKeys = set(cpdict.keys()).difference(MasterValue.keys())
					if DiffKeys:
						self.Logger("Warning: The key(s) {2} specified in '{0}.{3}' for '{1}' are not in that dictionary "\
						"in Parameters_Master.\nThis may lead to unexpected results.\n".format(Attr,Name,DiffKeys,VariableName), Print=True)
				if Name not in ParaDict: ParaDict[Name] = {}
				ParaDict[Name][VariableName] = NewVal

		if hasattr(Var,'Run'):
			if len(Var.Run)!=NbNames:
				self.Exit("Error: Number of entries for {}.Run not equal to {}.Names".format(Attr))
			for Name, flag in zip(Var.Name, Var.Run):
				if not flag:
					ParaDict.pop(Name)

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
