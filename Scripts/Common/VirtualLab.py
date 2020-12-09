#!/usr/bin/env python3

import sys
sys.dont_write_bytecode=True
import datetime
import os
import numpy as np
import shutil
import time
import inspect
import copy
from types import SimpleNamespace as Namespace
import contextlib
from importlib import import_module
import ast
import pickle
from multiprocessing import Process

import VLconfig
from Scripts.Common import Analytics, MPRun
from Scripts.Common.VLPackages import Salome, CodeAster
from Scripts.Common.VLTypes import Mesh as MeshFn, Sim as SimFn

class VLSetup():
	def __init__(self, Simulation, Project, StudyName, Parameters_Master, Parameters_Var=None, Mode='T'):
		# __force__ contains any keyword arguments passed using the -k argument when launching VirtualLab
		self.__force__ = self.GetArgParser()

		Simulation = self.__force__.get('Simulation',Simulation)
		Project = self.__force__.get('Project',Project)
		StudyName = self.__force__.get('StudyName',StudyName)
		Parameters_Master = self.__force__.get('Parameters_Master',Parameters_Master)
		Parameters_Var = self.__force__.get('Parameters_Var',Parameters_Var)
		Mode = self.__force__.get('Mode',Mode)

		# Set running mode
		if Mode.lower() in ('i', 'interactive'): self.mode = 'Interactive'
		elif Mode.lower() in ('t','terminal'): self.mode = 'Terminal'
		elif Mode.lower() in ('c', 'continuous'): self.mode = 'Continuous'
		elif Mode.lower() in ('h', 'headless'): self.mode = 'Headless'
		else : self.Exit("Error: Mode is not in; 'Interactive','Terminal','Continuous' or 'Headless'")

		self.Simulation = Simulation
		self.Project = Project
		self.StudyName = StudyName

		self.__ID__ = (datetime.datetime.now()).strftime("%y%m%d_%H%M%S")

		VL_DIR = VLconfig.VL_DIR
		if VL_DIR != sys.path[-1]: sys.path.pop(-1)

		### Define directories for VL from config file. If directory name doesn't start with '/'
		### it will be created relative to the TWD
		# Output directory - this is where meshes, Aster results and pre/post-processing will be stored.
		OUTPUT_DIR = getattr(VLconfig,'OutputDir', "{}/Output".format(VL_DIR))
		# Material directory
		MATERIAL_DIR = getattr(VLconfig,'MaterialDir', "{}/Materials".format(VL_DIR))
		# Input directory
		INPUT_DIR = getattr(VLconfig,'InputDir', "{}/Input".format(VL_DIR))

		# tmp directory
		TEMP_DIR = getattr(VLconfig,'TEMP_DIR',"/tmp")
		self.TMP_DIR = '{}/{}_{}'.format(TEMP_DIR, Project, self.__ID__)
		if Project == '.dev': self.TMP_DIR = "{}/dev".format(TEMP_DIR)

		self.MATERIAL_DIR = MATERIAL_DIR
		self.INPUT_DIR = '{}/{}/{}'.format(INPUT_DIR, Simulation, Project)

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

		# Output directory
		self.PROJECT_DIR = "{}/{}/{}".format(OUTPUT_DIR, Simulation, Project)
		self.STUDY_DIR = "{}/{}".format(self.PROJECT_DIR, StudyName)
		self.MESH_DIR = "{}/Meshes".format(self.PROJECT_DIR)

		self.Logger('### Launching VirtualLab ###',Print=True)

		# Create variables based on the namespaces (NS) in the Parameters file(s) provided
		self.NS = ['Mesh','Sim','ML']
		self.GetParams(Parameters_Master,Parameters_Var,self.NS)

		self.Salome = Salome.Salome(self, AddPath=[self.SIM_SCRIPTS])

		self.CodeAster = CodeAster.CodeAster(self)



	def Control(self, **kwargs):
		'''
		kwargs available:
		RunMesh: Boolean to dictate whether or not to create meshes
		RunSim: Boolean to dictate whether or not to run CodeAster
		'''
		kwargs.update(self.GetArgParser())

		# RunMesh = kwargs.get('RunMesh', True)
		# RunSim = kwargs.get('RunSim',True)
		# RunML = kwargs.get('RunML',True)

		sys.path = [self.COM_SCRIPTS,self.SIM_SCRIPTS] + sys.path

		MeshFn.Setup(self,**kwargs)

		SimFn.Setup(self,**kwargs)


		# # ML section
		# self.MLData = {}
		# if RunML and self.Parameters_Master.ML:
		# 	self.ML_DIR = "{}/ML".format(self.PROJECT_DIR)
		# 	os.makedirs(self.ML_DIR,exist_ok=True)
		# 	MLdicts = self.CreateParameters(self.Parameters_Master,self.Parameters_Var,'ML')
		# 	for Name, ParaDict in MLdicts.items():
		# 		self.MLData[Name] = Namespace(**ParaDict)

		'''
		# Gather information about what's running in VirtualLab, i.e. # Simulations, # Meshes
		NumSims = len(self.SimData)
		NumMeshes = len(set(MeshNames)) if NumSims else 0
		NumMeshesCr = len(self.MeshData)

		Infostr = "Simulation Type: {}\n"\
				  "Project: {}\n"\
				  "StudyName: {}\n"\
				  "Nb. Meshes: {}\n"\
				  "Nb. Simulations: {}".format(self.Simulation,self.Project,self.StudyName,NumMeshesCr,NumSims)
		self.Logger(Infostr)

		# Using inspect and ast we can get the name of the RunFile used and the
		# environment values which are used ()
		frame = inspect.stack()[1]
		RunFile = os.path.realpath(frame[0].f_code.co_filename)
		RunFileSC = inspect.getsource(inspect.getmodule(frame[0]))
		envdict = {'Simulation':self.Simulation,'Project':self.Project, \
					'StudyName':self.StudyName, 'Mode':self.mode, \
					'NumSims':NumSims,'NumMeshes':NumMeshes,'NumMeshesCr':NumMeshesCr}
		keywords = {'RunMesh':RunMesh,'RunSim':RunSim,'MeshCheck':None, \
					'ShowMesh':False, 'MeshThreads':1,'RunPreAster':True, \
					'RunAster':True, 'RunPostAster':True, 'ShowRes':True, \
					'SimThreads':1, 'mpi_nbcpu':1, 'mpi_nbnoeud':1, \
					'ncpus':1,'memory':2}
		for cd in ast.parse(RunFileSC).body:
			obj = getattr(cd,'value',None)
			fn = getattr(getattr(obj,'func',None),'attr',None)
			if fn in ('Mesh','Sim'):
				for kw in obj.keywords:
					if hasattr(kw.value,'value'): val=kw.value.value
					elif hasattr(kw.value,'n'): val=kw.value.n
					key = kw.arg
					if key == 'NumThreads':
						key = "MeshThreads" if fn == 'Mesh' else "SimThreads"
					keywords[key] = val

		envdict.update(keywords)

		# Function to analyse usage of VirtualLab to evidence impact for
		# use in future research grant applications. Can be turned off via
		# VLconfig.py. See Scripts/Common/Analytics.py for more details.
		if VLconfig.VL_ANALYTICS=="True": Analytics.event(envdict)

		'''

	def Mesh(self,**kwargs):
		return MeshFn.Run(self,**kwargs)

	def devMesh(self,**kwargs):
		return MeshFn.devRun(self,**kwargs)

	def Sim(self,**kwargs):
		return SimFn.Run(self,**kwargs)

	def devSim(self,**kwargs):
		return SimFn.devRun(self,**kwargs)

	def ML(self,**kwargs):
		for Name, MLdict in self.MLData.items():
			MLfn = import_module("ML.{}".format(MLdict.File))
			MLfn.main(self)


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
				self.LogFile = "{}/log/{}_{}.log".format(self.PROJECT_DIR,self.StudyName,self.__ID__)
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

		if os.path.isdir(self.TMP_DIR):
			if KeepDirs:
				kept = []
				for ct in os.listdir(self.TMP_DIR):
					SubDir = '{}/{}'.format(self.TMP_DIR,ct)
					if os.path.isdir(SubDir):
						if ct in KeepDirs: kept.append(SubDir)
						else : shutil.rmtree(SubDir)
				self.Logger("The following tmp directories have not been deleted:\n{}".format(kept),Print=True)
			else:
				shutil.rmtree(self.TMP_DIR)

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
		Master=getattr(Parameters_Master,Attr)
		Var=getattr(Parameters_Var,Attr)
		Names = getattr(Var, 'Name', [Master.Name])
		NbNames = len(Names)

		ParaDict = {Name:{} for Name in Names}

		for VariableName, MasterValue in Master.__dict__.items():
			# check types
			NewValues = getattr(Var, VariableName, [MasterValue]*NbNames)
			# Check the number of NewVals is correct
			if len(NewValues) != NbNames:
				self.Exit("Error: Number of entries for '{0}.{1}' not equal to '{0}.Names'".format(Attr,VariableName))
			for Name,NewVal in zip(Names,NewValues):
				if type(MasterValue)==dict:
					cpdict = copy.deepcopy(MasterValue)
					cpdict.update(NewVal)
					NewVal=cpdict
					DiffKeys = set(cpdict.keys()).difference(MasterValue.keys())
					if DiffKeys:
						self.Logger("Warning: The key(s) {2} specified in '{0}.{3}' for '{1}' are not in that dictionary "\
						"in Parameters_Master.\nThis may lead to unexpected results.\n".format(Attr,Name,DiffKeys,VariableName), Print=True)
				ParaDict[Name][VariableName] = NewVal

		if hasattr(Var,'Run'):
			if len(Var.Run)!=NbNames:
				self.Exit("Error: Number of entries for {}.Run not equal to {}.Names".format(Attr))
			for Name, flag in zip(Names, Var.Run):
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
