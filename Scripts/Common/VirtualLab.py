#!/usr/bin/env python3

import sys
sys.dont_write_bytecode=True
import datetime
import os
import numpy as np
import shutil
import time
from subprocess import Popen, PIPE, STDOUT
import inspect
import copy
from types import SimpleNamespace as Namespace
import contextlib
import VLconfig
from importlib import import_module
import ast
from Scripts.Common import Analytics, MPRun
from multiprocessing import Process
import pickle
import uuid

class VLSetup():
	def __init__(self, Simulation, Project, StudyName, Parameters_Master, Parameters_Var, Mode):

		# Set running mode
		if Mode in ('i', 'I', 'interactive', 'Interactive'): self.mode = 'Interactive'
		elif Mode in ('c', 'C', 'continuous', 'Continuous'): self.mode = 'Continuous'
		elif Mode in ('h', 'H', 'headless', 'Headless'): self.mode = 'Headless'
		else : self.Exit("Error: Mode is not in ('Interactive','Continuous','Headless')")

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

		# Code_Aster directory
		self.ASTER_DIR = VLconfig.ASTER_DIR

		# tmp directory
		TEMP_DIR = getattr(VLconfig,'TEMP_DIR',"/tmp")
		self.TMP_DIR = '{}/{}_{}'.format(TEMP_DIR, Project, self.__ID__)
		if Project == '.dev': self.TMP_DIR = "{}/dev".format(TEMP_DIR)

		# Define variables and run some checks
		# Script directories
		self.COM_SCRIPTS = "{}/Scripts/Common".format(VL_DIR)
		self.SIM_SCRIPTS = "{}/Scripts/{}".format(VL_DIR, Simulation)

		self.SIM_MESH = "{}/Mesh".format(self.SIM_SCRIPTS)
		self.SIM_PREASTER = "{}/PreAster".format(self.SIM_SCRIPTS)
		self.SIM_ASTER = "{}/Aster".format(self.SIM_SCRIPTS)
		self.SIM_POSTASTER = "{}/PostAster".format(self.SIM_SCRIPTS)

		# Materials directory
		self.MATERIAL_DIR = MATERIAL_DIR

		# Output directory
		self.PROJECT_DIR = "{}/{}/{}".format(OUTPUT_DIR, Simulation, Project)
		self.STUDY_DIR = "{}/{}".format(self.PROJECT_DIR, StudyName)
		self.MESH_DIR = "{}/Meshes".format(self.PROJECT_DIR)

		# Create dictionary of Parameters info
		self.Parameters = {'Master':Parameters_Master,'Var':Parameters_Var,'Dir':'{}/{}/{}'.format(INPUT_DIR, Simulation, Project)}

		self.Logger('### Launching VirtualLab ###',Print=True)

		self.Salome = VLSalome(self)

	def Control(self, **kwargs):
		'''
		kwargs available:
		RunMesh: Boolean to dictate whether or not to create meshes
		RunSim: Boolean to dictate whether or not to run CodeAster
		port: Give the port number of an open Salome instance to connect to
		'''
		RunMesh = kwargs.get('RunMesh', True)
		RunSim = kwargs.get('RunSim',True)

		sys.path.insert(0, self.COM_SCRIPTS)
		sys.path.insert(0, self.SIM_SCRIPTS)

		# Check the Parameter files exist
		self.ErrorCheck('Parameters')

		sys.path.insert(0, self.Parameters['Dir'])
		Main = __import__(self.Parameters['Master'])
		Var = __import__(self.Parameters['Var']) if self.Parameters['Var'] else None

		MainDict = copy.deepcopy(self.__dict__)
		MainDict.pop('Salome')

		MainMesh = getattr(Main, 'Mesh', None)
		MainSim = getattr(Main, 'Sim', None)

		self.Parameters_Master = Main

		# Create Mesh parameter files if they are required
		self.Meshes = {}
		if RunMesh and MainMesh:
			if not os.path.isdir(self.MESH_DIR): os.makedirs(self.MESH_DIR)
			self.GEOM_DIR = '{}/Geom'.format(self.TMP_DIR)
			if not os.path.isdir(self.GEOM_DIR): os.makedirs(self.GEOM_DIR)

			ParaMesh = getattr(Var, 'Mesh', None)
			MeshNames = getattr(ParaMesh, 'Name', [MainMesh.Name])
			NumMeshes = len(MeshNames)

			MeshDict = {MeshName:{} for MeshName in MeshNames}
			for VarName, Value in MainMesh.__dict__.items():
				NewVals = getattr(ParaMesh, VarName, False)
				# Check the number of NewVals is correct
				NumVals = len(NewVals) if NewVals else NumMeshes
				if NumVals != NumMeshes: self.Exit("Error: Number of entries for 'Mesh.{}' not equal to number of meshes".format(VarName))

				for i, MeshName in enumerate(MeshNames):
					if NewVals==False:
						Val = Value
					elif type(Value)==dict:
						Val = copy.deepcopy(Value)
						NV = NewVals[i]

						diff = set(NV.keys()).difference(Val)
						if diff:
							self.Logger("Warning: New key(s) {} specified in dictionary {} for mesh '{}'. This may lead to unexpected resutls".format(diff, VarName, MeshName), Print=True)

						Val.update(NV)
					else :
						Val = NewVals[i]
					MeshDict[MeshName][VarName] = Val

			if hasattr(ParaMesh,'Run'):
				if len(ParaMesh.Run)!=NumMeshes: self.Exit("Error: Number of entries for variable 'Mesh.Run' not equal to number of meshes")
				MeshNames = [mesh for mesh, flag in zip(MeshNames, ParaMesh.Run) if flag]

			sys.path.insert(0, self.SIM_MESH)
			for MeshName in MeshNames:
				ParaDict=MeshDict[MeshName]
				self.ErrorCheck('Mesh',MeshDict=ParaDict)
				self.WriteModule("{}/{}.py".format(self.GEOM_DIR, MeshName), ParaDict)
				self.Meshes[MeshName] = Namespace()
				self.Meshes[MeshName].__dict__.update(ParaDict)

		# Create Simulation parameter files
		self.Studies = {}
		if RunSim and MainSim:
			if not os.path.exists(self.ASTER_DIR):
				self.Exit("Error: CodeAster location invalid")

			if not os.path.isdir(self.STUDY_DIR): os.makedirs(self.STUDY_DIR)

			ParaSim = getattr(Var, 'Sim', None)
			SimNames = getattr(ParaSim, 'Name', [MainSim.Name])
			NumSims = len(SimNames)

			SimDict = {SimName:{} for SimName in SimNames}
			for VarName, Value in MainSim.__dict__.items():
				NewVals = getattr(ParaSim, VarName, False)
				# Check the number of NewVals is correct
				NumVals = len(NewVals) if NewVals else NumSims
				if NumVals!=NumSims: self.Exit("Error: Number of entries for 'Sim.{}' not equal to number of simulations".format(VarName))

				for i, SimName in enumerate(SimNames):
					if NewVals==False:
						Val = Value
					elif type(Value)==dict:
						Val = copy.deepcopy(Value)
						NV = NewVals[i]
						diff = set(NV.keys()).difference(Val)
						if diff:
							self.Logger("Warning: New key(s) {} specified in dictionary {} for sim '{}'. This may lead to unexpected resutls".format(diff, VarName, SimName),Print=True)

						Val.update(NV)
					else :
						Val = NewVals[i]
					SimDict[SimName][VarName] = Val

			if hasattr(ParaSim,'Run'):
				if len(ParaSim.Run)!=NumSims: self.Exit("Error: Number of entries for variable 'Sim.Run' not equal to number of simulations")
				SimNames = [sim for sim, flag in zip(SimNames, ParaSim.Run) if flag]

			MeshNames = []
			for SimName in SimNames:
				ParaDict = SimDict[SimName]
				self.ErrorCheck('Simulation',SimDict=ParaDict)
				StudyDict = {}
				# Define simulation related directories
				StudyDict['TMP_CALC_DIR'] = TMP_CALC_DIR = "{}/{}".format(self.TMP_DIR, SimName)
				StudyDict['CALC_DIR'] = CALC_DIR = "{}/{}".format(self.STUDY_DIR, SimName)
				if not os.path.isdir(TMP_CALC_DIR): os.makedirs(TMP_CALC_DIR)
				with open("{}/__init__.py".format(TMP_CALC_DIR),'w') as f: pass
				if not os.path.isdir(CALC_DIR): os.makedirs(CALC_DIR)

				StudyDict['PREASTER'] = "{}/PreAster".format(CALC_DIR)
				StudyDict['ASTER'] = "{}/Aster".format(CALC_DIR)
				StudyDict['POSTASTER'] = "{}/PostAster".format(CALC_DIR)
				StudyDict['MeshFile'] = "{}/{}.med".format(self.MESH_DIR, ParaDict['Mesh'])

				MeshNames.append(ParaDict['Mesh'])

				#Merge together Main and Study dict and write to file for salome/CodeAster to import
				MergeDict = {**MainDict, **StudyDict}
				self.WriteModule("{}/PathVL.py".format(TMP_CALC_DIR), MergeDict)
				# Write parameter file for salome/CodeAster to import
				self.WriteModule("{}/Parameters.py".format(TMP_CALC_DIR), ParaDict)
				shutil.copy("{}/Parameters.py".format(TMP_CALC_DIR), CALC_DIR)

				# Attach Parameters to StudyDict for ease of access
				StudyDict['Parameters'] = Namespace()
				StudyDict['Parameters'].__dict__.update(ParaDict)

				# Add StudyDict to Studies dictionary
				self.Studies[SimName] = StudyDict.copy()

		# Gather information about what's running in VirtualLab
		NumSims = len(self.Studies)
		NumMeshes = len(MeshNames) if NumSims else 0
		NumMeshesCr = len(self.Meshes)

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
					'StudyName':self.StudyName, 'Parameters_Master':self.Parameters['Master'], \
					'Parameters_Var':self.Parameters['Var'],'Mode':self.mode, \
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


	def Mesh(self, **kwargs):
		if not self.Meshes: return
		'''
		kwargs available:
		MeshCheck: input a meshname and it will open this mesh in the GUI
		ShowMesh: Opens up the meshes in the GUI. Boolean
		NumThreads: Number of different meshes to execute concurrently
		'''
		MeshCheck = kwargs.get('MeshCheck', None)
		ShowMesh = kwargs.get('ShowMesh', False)
		NumThreads = kwargs.get('NumThreads',1)

		# MeshCheck routine which allows you to mesh in the GUI (Used for debugging).
		# The script will terminate after this routine
		if MeshCheck and MeshCheck in self.Meshes.keys():
			self.Logger('### Meshing {} in GUI ###\n'.format(MeshCheck), Print=True)

			MeshParaFile = "{}/{}.py".format(self.GEOM_DIR,MeshCheck)
			MeshScript = "{}/{}.py".format(self.SIM_MESH, self.Meshes[MeshCheck].File)
			# The file MeshParaFile is passed to MeshScript to create the mesh in the GUI
			self.Salome.Run(MeshScript, ArgList=[MeshParaFile], GUI=True)
			self.Exit('Terminating after checking mesh')

		elif MeshCheck and MeshCheck not in self.Meshes.keys():
			self.Exit("Error: '{}' specified for MeshCheck is not one of meshes to be created.\n"\
					  "Meshes to be created are:{}".format(MeshCheck, list(self.Meshes.keys())))

		self.Logger('\n### Starting Meshing ###\n',Print=True)

		NumMeshes = len(self.Meshes)
		NumThreads = min(NumThreads,NumMeshes)
		MeshError = []

		# Start #NumThreads number of Salome sessions
		Ports = self.Salome.Start(NumThreads, OutFile=self.LogFile)
		PortCount = {Port:0 for Port in Ports}

		# Script which is used to import the necessary mesh function
		MeshScript = '{}/MeshRun.py'.format(self.COM_SCRIPTS)
		AddPath = [self.SIM_MESH, self.GEOM_DIR]
		ArgDict = {}
		if os.path.isfile('{}/config.py'.format(self.SIM_MESH)): ArgDict["ConfigFile"] = True

		tmpLogstr = "" if self.mode=='Interactive' else "{}/{}_log"
		MeshStat = {}
		NumActive=NumComplete=0
		SalomeReset = 400 #Close Salome session(s) & open new after this many meshes due to memory leak
		for MeshName, MeshPara in self.Meshes.items():
			self.Logger("'{}' started".format(MeshName),Print=True)

			port = Ports.pop(0)
			tmpLog = tmpLogstr.format(self.GEOM_DIR,MeshName)

			ArgDict.update(Name=MeshName, MESH_FILE="{}/{}.med".format(self.MESH_DIR, MeshName),
						   RCfile="{}/{}_RC.txt".format(self.GEOM_DIR,MeshName))

			Proc = self.Salome.Run(MeshScript, Port=port, AddPath=AddPath, ArgDict=ArgDict, OutFile=tmpLog)
			MeshStat[MeshName] = [Proc,port]
			PortCount[port] +=1
			NumActive+=1
			NumComplete+=1
			while NumActive==NumThreads or NumComplete==NumMeshes:
				for tmpMeshName, SalomeInfo in MeshStat.copy().items():
					Proc, port = SalomeInfo
					Poll = Proc.poll()
					# If SubProc finished Poll will change from None to errorcode
					if Poll is not None:
						if self.mode != 'Interactive':
							with open(tmpLogstr.format(self.GEOM_DIR,tmpMeshName),'r') as rtmpLog:
								self.Logger("\nOutput for '{}':\n{}".format(tmpMeshName,rtmpLog.read()))

						# Check if any returncode provided
						RCfile="{}/{}_RC.txt".format(self.GEOM_DIR,tmpMeshName)
						if os.path.isfile(RCfile):
							with open(RCfile,'r') as f:
								returncode=int(f.readline())
							AffectedSims = [Name for Name, StudyDict in self.Studies.items() if StudyDict["Parameters"].Mesh == tmpMeshName]
							MeshPara = self.Meshes[tmpMeshName]
							MeshImp = import_module('Mesh.{}'.format(MeshPara.File))
							# Check in meshfile for error code handling
							if hasattr(MeshImp,'HandleRC'):
								self.Logger("'{}'' returned code {}. "\
											"Passed to HandleRC function.".format(tmpMeshName,returncode),Print=True)
								MeshImp.HandleRC(returncode,self.Studies,AffectedSims,tmpMeshName, MeshError)
							else :
								self.Logger("'{}' returned code {}. Added to error list "\
											"since no HandleRC function found".format(tmpMeshName,returncode),Print=True)
								MeshError.append(tmpMeshName)
						# SubProc returned with error code
						elif Poll != 0:
							self.Logger("'{}' finished with errors".format(tmpMeshName), Print=True)
							MeshError.append(tmpMeshName)
						# SubProc returned successfully
						else :
							self.Logger("'{}' completed successfully".format(tmpMeshName), Print=True)
							shutil.copy("{}/{}.py".format(self.GEOM_DIR,tmpMeshName), self.MESH_DIR)

						# Check if a new salome sesion is needed to free up memory
						# for the next mesh
						if NumComplete < NumMeshes and PortCount[port] >= SalomeReset/NumThreads:
							self.Logger("Limit reached on Salome session {}".format(port))
							Salome_Close = self.Salome.Close(port)
							port = self.Salome.Start(OutFile=self.LogFile)[0]
							PortCount[port] = 0
							Salome_Close.wait()

						MeshStat.pop(tmpMeshName)
						Proc.terminate()
						NumActive-=1
						Ports.append(port)

				time.sleep(0.1)
				if not len(MeshStat): break

		if MeshError: self.Exit("The following Meshes finished with errors:\n{}".format(MeshError),KeepDirs=['Geom'])

		self.Logger('\n### Meshing Completed ###',Print=True)
		if ShowMesh:
			self.Logger("Opening mesh files in Salome",Print=True)
			MeshPaths = ["{}/{}.med".format(self.MESH_DIR, name) for name in self.Meshes.keys()]
			SubProc = Popen('salome {}/ShowMesh.py args:{} '.format(self.COM_SCRIPTS,",".join(MeshPaths)), shell='TRUE')
			SubProc.wait()
			self.Exit("Terminating after mesh viewing")

	def Sim(self, **kwargs):
		if not self.Studies: return

		'''
		kwargs
		### PreAster kwargs ###
		RunPreAster: Run PreAster calculations. Boolean

		### Aster kwargs ###
		RunAster: Run CodeAster. Boolean
		mpi_nbcpu: Num CPUs for parallel CodeAster. Only available if code aster compiled for parallelism.
		mpi_nbnoeud: Num Nodes for parallel CodeAster. Only available if code aster compiled for parallelism.
		ncpus: Number of CPUs for regular CodeAster
		memory: Amount of memory (Gb) allocated to CodeAster

		### PostAster kwargs ###
		RunPostAster: Run PostAster calculations. Boolean
		ShowRes: Opens up all results files in Salome GUI. Boolean

		'''
		RunPreAster = kwargs.get('RunPreAster',True)
		RunAster = kwargs.get('RunAster', True)
		RunPostAster = kwargs.get('RunPostAster', True)
		ShowRes = kwargs.get('ShowRes', False)
		mpi_nbcpu = kwargs.get('mpi_nbcpu',1)
		mpi_nbnoeud = kwargs.get('mpi_nbnoeud',1)
		ncpus = kwargs.get('ncpus',1)
		memory = kwargs.get('memory',2)
		NumThreads = kwargs.get('NumThreads',1)

		self.Logger('\n### Starting Simulations ###\n', Print=True)

		NumSim = len(self.Studies)
		NumThreads = min(NumThreads,NumSim)

		SimLogFile = "{}/Output.log"

		SimMaster = self.Parameters_Master.Sim
		if RunPreAster and hasattr(SimMaster,'PreAsterFile'):
			self.Logger('>>> PreAster Stage', Print=True)
			sys.path.insert(0, self.SIM_PREASTER)

			count, NumActive = 0, 0
			PreError = []
			PreStat = {}
			for Name, StudyDict in self.Studies.items():
				PreAsterFile = StudyDict['Parameters'].PreAsterFile
				if not PreAsterFile: continue
				PreAster = import_module(PreAsterFile)
				PreAsterSgl = getattr(PreAster, 'Single',None)
				if not PreAsterSgl: continue


				self.Logger("Pre-Aster for '{}' started\n".format(Name),Print=True)
				if not os.path.isdir(StudyDict['PREASTER']): os.makedirs(StudyDict['PREASTER'])

				proc = Process(target=MPRun.main, args=(self,StudyDict,PreAsterSgl))

				if self.mode == 'Interactive':
					proc.start()
				else :
					with open(SimLogFile.format(StudyDict['PREASTER']), 'w') as f:
						with contextlib.redirect_stdout(f):
							# stderr may need to be written to a seperate file and then copied over
							with contextlib.redirect_stderr(sys.stdout):
								proc.start()

				count +=1
				NumActive +=1
				PreStat[Name] = proc
				while NumActive==NumThreads or count==NumSim:
					for tmpName, proc in PreStat.copy().items():
						EC = proc.exitcode
						if EC == None:
							continue
						tmpStudyDict = self.Studies[tmpName]
						if EC == 0:
							self.Logger("Pre-Aster for '{}' completed\n".format(tmpName),Print=True)
						else :
							self.Logger("Pre-Aster for '{}' returned error code {}\n".format(tmpName,EC),Print=True)
							PreError.append(tmpName)

						if self.mode != 'Interactive':
							self.Logger("See {} for details".format(SimLogFile.format(tmpStudyDict['PREASTER'])),Print=EC)

						PreStat.pop(tmpName)
						NumActive-=1

						picklefile = "{}/StudyDict.pickle".format(tmpStudyDict["TMP_CALC_DIR"])
						if os.path.isfile(picklefile):
							with open(picklefile, 'rb') as handle:
								NewDict = pickle.load(handle)
							tmpStudyDict.update(NewDict)
							os.remove(picklefile)

					time.sleep(0.1)
					if not len(PreStat): break

			if PreError: self.Exit("The following PreAster routine(s) finished with errors:\n{}".format(PreError),KeepDirs=PreError)

			PreAster = import_module(SimMaster.PreAsterFile)
			if hasattr(PreAster, 'Combined'):
				self.Logger('\n Running PreAster Combined #\n', Print=True)
				PreAster.Combined(self)

			self.Logger('>>> PreAster Stage Complete\n', Print=True)

		if RunAster and hasattr(SimMaster,'AsterFile'):
			self.Logger('>>> Aster Stage', Print=True)
			AsterError = []
			AsterStat = {}
			count, NumActive = 0, 0
			for Name, StudyDict in self.Studies.items():
				if not os.path.isdir(StudyDict['ASTER']): os.makedirs(StudyDict['ASTER'])

				# Define location of export and Aster file
				asterfile = '{}/{}.comm'.format(self.SIM_ASTER,StudyDict['Parameters'].AsterFile)
				exportfile = "{}/Export".format(StudyDict['ASTER'])

				# Create export file and write to file
				exportstr = 'P actions make_etude\n' + \
				'P mode batch\n' + \
				'P version stable\n' + \
				'P time_limit 99999\n' + \
				'P mpi_nbcpu {}\n'.format(mpi_nbcpu) + \
				'P mpi_nbnoeud {}\n'.format(mpi_nbnoeud) + \
				'P ncpus {}\n'.format(ncpus) + \
				'P memory_limit {!s}\n'.format(float(1024*memory)) +\
				'F mmed {} D  20\n'.format(StudyDict["MeshFile"]) + \
				'F comm {} D  1\n'.format(asterfile) + \
				'F mess {}/AsterLog R  6\n'.format(StudyDict['ASTER']) + \
				'R repe {} R  0\n'.format(StudyDict['ASTER'])
				with open(exportfile,'w+') as e:
					e.write(exportstr)

				self.Logger("Aster for '{}' started".format(Name),Print=True)
				AsterStat[Name] = self.AsterExec(StudyDict, exportfile)
				count +=1
				NumActive +=1

				while NumActive==NumThreads or count==NumSim:
					for tmpName, Proc in AsterStat.copy().items():
						Poll = Proc.poll()
						if Poll == None:
							continue
						tmpStudyDict = self.Studies[tmpName]
						if self.mode == 'Interactive':
							with open('{}/Aster.txt'.format(tmpStudyDict['TMP_CALC_DIR']),'r') as f:
								EC = int(f.readline())
						else : EC = Poll
						# elif self.mode == 'Continuous':
						# 	os.remove('{}/ContinuousAsterLog'.format(tmpStudyDict['ASTER']))

						if EC == 0:
							self.Logger("Aster for '{}' completed".format(tmpName),Print=True)
						else :
							self.Logger("Aster for '{}' returned error code {}.".format(tmpName,EC))
							AsterError.append(tmpName)

						if self.mode != 'Interactive':
							self.Logger("See {}/AsterLog for details".format(tmpStudyDict['ASTER']),Print=EC)

						AsterStat.pop(tmpName)
						Proc.terminate()

					if not len(AsterStat): break
					time.sleep(0.1)

			if AsterError: self.Exit("The following simulation(s) finished with errors:\n{}".format(AsterError),KeepDirs=AsterError)

			self.Logger('>>> Aster Stage Complete\n', Print=True)

		if RunPostAster and hasattr(SimMaster,'PostAsterFile'):
			sys.path.insert(0, self.SIM_POSTASTER)
			self.Logger('>>> PostAster Stage', Print=True)

			count, NumActive = 0, 0
			PostError = []
			PostStat = {}
			for Name, StudyDict in self.Studies.items():
				PostAsterFile = getattr(StudyDict['Parameters'],'PostAsterFile', None)
				if not PostAsterFile : continue
				PostAster = import_module(PostAsterFile)
				PostAsterSgl = getattr(PostAster, 'Single',None)
				if not PostAsterSgl: continue

				self.Logger("PostAster for '{}' started".format(Name),Print=True)
				if not os.path.isdir(StudyDict['POSTASTER']): os.makedirs(StudyDict['POSTASTER'])

				proc = Process(target=MPRun.main, args=(self,StudyDict,PostAsterSgl))
				if self.mode == 'Interactive':
					proc.start()
				else :
					with open(SimLogFile.format(StudyDict['POSTASTER']), 'w') as f:
						with contextlib.redirect_stdout(f):
							# stderr may need to be written to a seperate file and then copied over
							with contextlib.redirect_stderr(sys.stdout):
								proc.start()

				count +=1
				NumActive +=1
				PostStat[Name] = proc
				while NumActive==NumThreads or count==NumSim:
					for tmpName, proc in PostStat.copy().items():
						EC = proc.exitcode
						if EC == None:
							continue

						tmpStudyDict = self.Studies[tmpName]
						if EC == 0:
							self.Logger("Post-Aster for '{}' completed".format(tmpName),Print=True)
						else :
							self.Logger("Post-Aster for '{}' returned error code {}".format(tmpName,EC),Print=True)
							PostError.append(tmpName)

						if self.mode != 'Interactive':
							self.Logger("See {} for details".format(SimLogFile.format(tmpStudyDict['POSTASTER'])),Print=EC)

						PostStat.pop(tmpName)
						NumActive-=1

						picklefile = "{}/StudyDict.pickle".format(tmpStudyDict["TMP_CALC_DIR"])
						if os.path.isfile(picklefile):
							with open(picklefile, 'rb') as handle:
								NewDict = pickle.load(handle)
							tmpStudyDict.update(NewDict)
							os.remove(picklefile)

					time.sleep(0.1)
					if not len(PostStat): break

			if PostError: self.Exit("The following PostAster routine(s) finished with errors:\n{}".format(PostError), KeepDirs=PostError)

			PostAster = import_module(SimMaster.PostAsterFile)
			if hasattr(PostAster, 'Combined'):
				self.Logger('Combined function started', Print=True)

				if self.mode == 'Interactive':
					PostAster.Combined(self)
				else :
					with open(self.LogFile, 'a') as f:
						with contextlib.redirect_stdout(f):
							# stderr may need to be written to a seperate file and then copied over
							with contextlib.redirect_stderr(sys.stdout):
								PostAster.Combined(self)

				self.Logger('Combined function complete', Print=True)

			self.Logger('>>> PostAster Stage Complete\n', Print=True)

		self.Logger('### Simulations Completed ###',Print=True)

		# Opens up all results in ParaVis
		if ShowRes:
			print("### Opening .rmed files in ParaVis ###\n")
			ResFiles = {}
			for SimName, StudyDict in self.Studies.items():
				for root, dirs, files in os.walk(StudyDict['CALC_DIR']):
					for file in files:
						fname, ext = os.path.splitext(file)
						if ext == '.rmed':
							ResFiles["{}_{}".format(SimName,fname)] = "{}/{}".format(root, file)
			Script = "{}/ShowRes.py".format(self.COM_SCRIPTS)
			SubProc = self.Salome.Run(Script, GUI=True, ArgDict=ResFiles)
			SubProc.wait()

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

	def AsterExec(self, StudyDict, exportfile, **kwargs):
		AddPath = kwargs.get('AddPath',[])

		AddPath = [AddPath] if type(AddPath) == str else AddPath
		AddPath += [self.COM_SCRIPTS, self.TMP_DIR, StudyDict['TMP_CALC_DIR']]
		PythonPath = ["PYTHONPATH={}:$PYTHONPATH;".format(path) for path in AddPath]
		PreCond = PythonPath + ["export PYTHONPATH;export PYTHONDONTWRITEBYTECODE=1;"]
		PreCond = ''.join(PreCond)

		# Create different command file depending on the mode
		errfile = '{}/Aster.txt'.format(StudyDict['TMP_CALC_DIR'])
		if self.mode == 'Interactive':
			xtermset = "-hold -T 'Study: {}' -sb -si -sl 2000".format(StudyDict["Parameters"].Name)
			command = "xterm {} -e '{} {}; echo $? >'{}".format(xtermset, self.ASTER_DIR, exportfile, errfile)
		elif self.mode == 'Continuous':
			command = "{} {} > {}/Output.log ".format(self.ASTER_DIR, exportfile, StudyDict['ASTER'])
		else :
			command = "{} {} >/dev/null 2>&1".format(self.ASTER_DIR, exportfile)

		# Start Aster subprocess
		proc = Popen(PreCond + command , shell='TRUE')
		return proc

	def Logger(self,Text='',**kwargs):
		Prnt = kwargs.get('Print',False)

		if not hasattr(self,'LogFile'):
			print(Text)
			if self.mode=='Interactive':
				self.LogFile = None
			else:
				self.LogFile = "{}/log/{}_{}.log".format(self.PROJECT_DIR,self.StudyName,self.__ID__)
				os.makedirs(os.path.dirname(self.LogFile), exist_ok=True)
				with open(self.LogFile,'w') as f:
					f.write(Text)
				print("Detailed output written to {}".format(self.LogFile))
			return

		if self.mode=='Interactive':
			print(Text)
		else:
			if Prnt: print(Text)
			with open(self.LogFile,'a') as f:
				f.write(Text+"\n")



	def ErrorCheck(self, Stage, **kwargs):
		if Stage == 'Parameters':
			# Check if the Input directory exists
			if not os.path.isdir(self.Parameters['Dir']):
				self.Exit("Directory '{}' does not exist".format(self.Parameters['Dir']))

			# Check 'Parameters_Master' exists
			if not os.path.exists('{}/{}.py'.format(self.Parameters['Dir'], self.Parameters['Master'])):
				self.Exit("Parameters_Master file '{}' not in directory {}".format(self.Parameters['Master'], self.Parameters['Dir']))

			# Check that 'Parameters_Var' exists (if not None)
			if self.Parameters['Var'] and not os.path.exists('{}/{}.py'.format(self.Parameters['Dir'], self.Parameters['Var'])):
				self.Exit("Parameters_Var file '{}' not in  directory {}".format(self.Parameters['Var'],self.Parameters['Dir']))

		if Stage == 'Mesh':
			MeshDict = kwargs.get('MeshDict')
			if os.path.exists('{}/{}.py'.format(self.SIM_MESH,MeshDict['File'])):
				## import Mesh
				## MeshFile = getattr(Mesh, MeshDict['File'])
				MeshFile = __import__(MeshDict['File'])
				ErrorFunc = getattr(MeshFile, 'GeomError', None)
				if ErrorFunc:
					ParamMesh = Namespace()
					ParamMesh.__dict__.update(MeshDict)
					err = ErrorFunc(ParamMesh)
				else : err = None
				if err: self.Exit("GeomError in '{}' - {}".format(MeshDict['Name'], err))
			else:
				self.Exit("Mesh file '{}' does not exist in {}".format(MeshDict['File'], self.SIM_MESH))

		if Stage == 'Simulation':
			SimDict = kwargs.get('SimDict')
			# Check that the scripts provided exist
			if 'PreAsterFile' in SimDict:
				if not os.path.isfile('{}/{}.py'.format(self.SIM_PREASTER,SimDict['PreAsterFile'])):
					self.Exit("PreAsterFile '{}' not in directory '{}'".format(SimDict['PreAsterFile'], self.SIM_PREASTER))
			if not os.path.isfile('{}/{}.comm'.format(self.SIM_ASTER,SimDict['AsterFile'])):
				self.Exit("AsterFile '{}' not in directory '{}'".format(SimDict['AsterFile'], self.SIM_ASTER))
			if 'PostAsterFile' in SimDict:
				if not os.path.isfile('{}/{}.py'.format(self.SIM_POSTASTER,SimDict['PostAsterFile'])):
					self.Exit("PostAsterFile '{}' not in directory '{}'".format(SimDict['PostAsterFile'], self.SIM_POSTASTER))

			# Check either the mesh is in the mesh directory or that it is a mesh to be created
			if SimDict['Mesh'] in (getattr(self, 'Meshes', {})).keys(): pass
			elif os.path.isfile("{}/{}.med".format(self.MESH_DIR, SimDict['Mesh'])): pass
			else : self.Exit("Mesh '{}' isn't being created and is not in the mesh directory '{}'".format(SimDict['Mesh'], self.MESH_DIR))

			Materials = SimDict.get('Materials',[])
			if type(Materials)==str: Materials = [Materials]
			elif type(Materials)==dict:Materials = Materials.values()
			for mat in set(Materials):
				if not os.path.exists('{}/{}'.format(self.MATERIAL_DIR, mat)):
					self.Exit("Material '{}' isn't in the materials directory '{}'".format(mat, self.MATERIAL_DIR))

	def Exit(self,mess='',KeepDirs=[]):
		self.Logger(mess, Print=True)
		self.Cleanup(KeepDirs)
		sys.exit()

	def Cleanup(self,KeepDirs=[]):
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

class VLSalome():
	def __init__(self, super):
		self.TMP_DIR = super.TMP_DIR
		self.COM_SCRIPTS = super.COM_SCRIPTS
		self.SIM_SCRIPTS = super.SIM_SCRIPTS
		self.Logger = super.Logger
		self.Ports = []
		self.LogFile = super.LogFile


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
			SubProc.wait()
			if SubProc.returncode != 0:
				self.Logger("Error during Salome initiation",Print=True)
				return False

			with open(portfile,'r') as f:
				port = int(f.readline())
			NewPorts.append(port)

		self.Logger('Salome opened on port(s) {}\n'.format(NewPorts))
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

		if not self.Ports:
			self.Start()

		Port = kwargs.get('Port', self.Ports[0])

		OutFile = ErrFile = kwargs.get('OutFile', self.LogFile)
		ErrFile = kwargs.get('ErrFile',ErrFile)

		output = ''
		if OutFile: output += " >>{}".format(OutFile)
		if ErrFile: output += " 2>>{}".format(ErrFile)

		command = "salome shell -p{!s} {} args:{} {}".format(Port, Script, Args, output)

		SubProc = Popen(PythonPath + command, shell='TRUE')
		return SubProc

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
