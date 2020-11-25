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
from importlib import import_module
import ast
import pickle
from multiprocessing import Process

import VLconfig
from Scripts.Common import Analytics, MPRun
from .VLPackages import Salome, CodeAster

class VLSetup():
	def __init__(self, Simulation, Project, StudyName, Parameters_Master, Parameters_Var, Mode):
		# __force__ contains any keyword arguments passed using the -k argument when launching VirtualLab
		self.__force__ = self.__ForceArgs__(sys.argv[1:])

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
		# Code_Aster directory
		self.ASTER_DIR = VLconfig.ASTER_DIR

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
		NS = ['Mesh','Sim','ML']
		self.GetParams(Parameters_Master,Parameters_Var,NS)

		self.Salome = Salome.Salome(self, AddPath=[self.SIM_SCRIPTS])

		self.CodeAster = CodeAster.CodeAster(self)



	def Control(self, **kwargs):
		'''
		kwargs available:
		RunMesh: Boolean to dictate whether or not to create meshes
		RunSim: Boolean to dictate whether or not to run CodeAster
		port: Give the port number of an open Salome instance to connect to
		'''
		kwargs.update(self.__force__)

		RunMesh = kwargs.get('RunMesh', True)
		RunSim = kwargs.get('RunSim',True)
		RunML = kwargs.get('RunML',True)

		sys.path.insert(0, self.COM_SCRIPTS)
		sys.path.insert(0, self.SIM_SCRIPTS)

		# Meta information about the class which will be passed to CodeAster
		MetaInfo = {key:val for key,val in self.__dict__.items() if type(val)==str}

		# Create MeshData which contains all of the mesh related information
		self.MeshData = {}
		if RunMesh and self.Parameters_Master.Mesh:
			self.GEOM_DIR = '{}/Geom'.format(self.TMP_DIR)
			os.makedirs(self.MESH_DIR, exist_ok=True)
			os.makedirs(self.GEOM_DIR, exist_ok=True)
			# Get dictionary of mesh parameters using Parameters_Master and Parameters_Var
			MeshDicts = self.CreateParameters(self.Parameters_Master,self.Parameters_Var,'Mesh')
			sys.path.insert(0, self.SIM_MESH)

			for MeshName, ParaDict in MeshDicts.items():
				# Run checks
				# Check that mesh file exists
				if not os.path.exists('{}/{}.py'.format(self.SIM_MESH,ParaDict['File'])):
					self.Exit("Mesh file '{}' does not exist in {}".format(ParaDict['File'], self.SIM_MESH))

				MeshFile = import_module(ParaDict['File'])
				try :
					err = MeshFile.GeomError(Namespace(**ParaDict))
					if err: self.Exit("GeomError in '{}' - {}".format(MeshDict['Name'], err))
				except AttributeError:
					pass
				# Checks complete

				self.WriteModule("{}/{}.py".format(self.GEOM_DIR, MeshName), ParaDict)
				self.MeshData[MeshName] = Namespace(**ParaDict)

		# Create SimData which contains all of the mesh related information
		self.SimData = {}
		if RunSim and self.Parameters_Master.Sim:
			# Check that Code Aster exists in the specified lcoation
			# if not os.path.exists(self.ASTER_DIR):
			# 	self.Exit("Error: CodeAster location invalid")
			os.makedirs(self.STUDY_DIR, exist_ok=True)

			SimDicts = self.CreateParameters(self.Parameters_Master,self.Parameters_Var,'Sim')
			MeshNames = []
			for SimName, ParaDict in SimDicts.items():
				# Run checks
				# Check files exist
				if not self.__CheckFile__(self.SIM_PREASTER,ParaDict.get('PreAsterFile'),'py'):
					self.Exit("PreAsterFile '{}.py' not in directory {}".format(ParaDict['PreAsterFile'],self.SIM_PREASTER))
				if not self.__CheckFile__(self.SIM_ASTER,ParaDict.get('AsterFile'),'comm'):
					self.Exit("AsterFile '{}.comm' not in directory {}".format(ParaDict['AsterFile'],self.SIM_ASTER,))
				if not self.__CheckFile__(self.SIM_POSTASTER, ParaDict.get('PostAsterFile'), 'py'):
					self.Exit("PostAsterFile '{}.py' not in directory {}".format(ParaDict['PostAsterFile'],self.SIM_POSTASTER))
				# Check mesh will be available
				if not (ParaDict['Mesh'] in self.MeshData or self.__CheckFile__(self.MESH_DIR, ParaDict['Mesh'], 'med')):
					self.Exit("Mesh '{}' isn't being created and is not in the mesh directory '{}'".format(ParaDict['Mesh'], self.MESH_DIR))
				# Check materials used
				Materials = ParaDict.get('Materials',[])
				if type(Materials)==str: Materials = [Materials]
				elif type(Materials)==dict: Materials = Materials.values()
				MatErr = [mat for mat in set(Materials) if not os.path.isdir('{}/{}'.format(self.MATERIAL_DIR, mat))]
				if MatErr:
						self.Exit("Material(s) {} specified for {} not available.\n"\
						"Please see the materials directory {} for options.".format(MatErr,SimName,self.MATERIAL_DIR))
				# Checks complete

				# Create dict of simulation specific information to be nested in SimData
				StudyDict = {}
				StudyDict['TMP_CALC_DIR'] = TMP_CALC_DIR = "{}/{}".format(self.TMP_DIR, SimName)
				StudyDict['CALC_DIR'] = CALC_DIR = "{}/{}".format(self.STUDY_DIR, SimName)
				StudyDict['PREASTER'] = "{}/PreAster".format(CALC_DIR)
				StudyDict['ASTER'] = "{}/Aster".format(CALC_DIR)
				StudyDict['POSTASTER'] = "{}/PostAster".format(CALC_DIR)
				StudyDict['MeshFile'] = "{}/{}.med".format(self.MESH_DIR, ParaDict['Mesh'])

				# Create tmp directory & add __init__ file so that it can be treated as a package
				if not os.path.isdir(TMP_CALC_DIR): os.makedirs(TMP_CALC_DIR)
				with open("{}/__init__.py".format(TMP_CALC_DIR),'w') as f: pass
				# Combine Meta information with that from Study dict and write to file for salome/CodeAster to import
				self.WriteModule("{}/PathVL.py".format(TMP_CALC_DIR), {**MetaInfo, **StudyDict})
				# Write Sim Parameters to file for Salome/CodeAster to import
				self.WriteModule("{}/Parameters.py".format(TMP_CALC_DIR), ParaDict)

				# Attach Parameters to StudyDict for ease of access
				StudyDict['Parameters'] = Namespace(**ParaDict)
				# Add StudyDict to SimData dictionary
				self.SimData[SimName] = StudyDict.copy()

				MeshNames.append(ParaDict['Mesh'])

		# ML section
		self.MLData = {}
		if RunML and self.Parameters_Master.ML:
			self.ML_DIR = "{}/ML".format(self.PROJECT_DIR)
			os.makedirs(self.ML_DIR,exist_ok=True)
			MLdicts = self.CreateParameters(self.Parameters_Master,self.Parameters_Var,'ML')
			for Name, ParaDict in MLdicts.items():
				self.MLData[Name] = Namespace(**ParaDict)


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


	def Mesh(self, **kwargs):
		if not self.MeshData: return
		kwargs.update(self.__force__)
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
		if MeshCheck and MeshCheck in self.MeshData.keys():
			self.Logger('### Meshing {} in GUI ###\n'.format(MeshCheck), Print=True)

			MeshParaFile = "{}/{}.py".format(self.GEOM_DIR,MeshCheck)
			MeshScript = "{}/{}.py".format(self.SIM_MESH, self.MeshData[MeshCheck].File)
			# The file MeshParaFile is passed to MeshScript to create the mesh in the GUI
			SubProc = self.Salome.Run(MeshScript, ArgList=[MeshParaFile], GUI=True)
			SubProc.wait()
			self.Exit('Terminating after checking mesh')

		elif MeshCheck and MeshCheck not in self.MeshData.keys():
			self.Exit("Error: '{}' specified for MeshCheck is not one of meshes to be created.\n"\
					  "Meshes to be created are:{}".format(MeshCheck, list(self.MeshData.keys())))

		self.Logger('\n### Starting Meshing ###\n',Print=True)

		NumMeshes = len(self.MeshData)
		NumThreads = min(NumThreads,NumMeshes)
		MeshError = []

		# Start #NumThreads number of Salome sessions
		Ports = self.Salome.Start(NumThreads, OutFile=self.LogFile)
		# Exit if no Salome sessions have been created
		if len(Ports)==0:
			self.Exit("Salome not initiated",Print=True)
		# Reduce NumThreads if fewer salome sessions have been created than requested
		elif len(Ports) < NumThreads:
			NumThreads=len(Ports)

		# Keep count number of meshes each session has created due to memory leak
		PortCount = {Port:0 for Port in Ports}

		# Script which is used to import the necessary mesh function
		MeshScript = '{}/MeshRun.py'.format(self.COM_SCRIPTS)
		AddPath = [self.SIM_MESH, self.GEOM_DIR]
		ArgDict = {}
		if os.path.isfile('{}/config.py'.format(self.SIM_MESH)): ArgDict["ConfigFile"] = True

		tmpLogstr = "" if self.mode in ('Interactive','Terminal') else "{}/{}_log"
		MeshStat = {}
		NumActive=NumComplete=0
		SalomeReset = 500 #Close Salome session(s) & open new after this many meshes due to memory leak
		for MeshName, MeshPara in self.MeshData.items():
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
						if self.mode not in ('Interactive','Terminal'):
							with open(tmpLogstr.format(self.GEOM_DIR,tmpMeshName),'r') as rtmpLog:
								self.Logger("\nOutput for '{}':\n{}".format(tmpMeshName,rtmpLog.read()))

						# Check if any returncode provided
						RCfile="{}/{}_RC.txt".format(self.GEOM_DIR,tmpMeshName)
						if os.path.isfile(RCfile):
							with open(RCfile,'r') as f:
								returncode=int(f.readline())
							AffectedSims = [Name for Name, StudyDict in self.SimData.items() if StudyDict["Parameters"].Mesh == tmpMeshName]
							MeshPara = self.MeshData[tmpMeshName]
							MeshImp = import_module('Mesh.{}'.format(MeshPara.File))
							# Check in meshfile for error code handling
							if hasattr(MeshImp,'HandleRC'):
								self.Logger("'{}'' returned code {}. "\
											"Passed to HandleRC function.".format(tmpMeshName,returncode),Print=True)
								MeshImp.HandleRC(returncode,self.SimData,AffectedSims,tmpMeshName, MeshError)
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
			MeshPaths = ["{}/{}.med".format(self.MESH_DIR, name) for name in self.MeshData.keys()]
			SubProc = Popen('salome {}/ShowMesh.py args:{} '.format(self.COM_SCRIPTS,",".join(MeshPaths)), shell='TRUE')
			SubProc.wait()
			self.Exit("Terminating after mesh viewing")

	def Sim(self, **kwargs):
		if not self.SimData: return
		kwargs.update(self.__force__)
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

		NumSim = len(self.SimData)
		NumThreads = min(NumThreads,NumSim)

		SimLogFile = "{}/Output.log"

		SimMaster = self.Parameters_Master.Sim
		if RunPreAster and hasattr(SimMaster,'PreAsterFile'):
			self.Logger('>>> PreAster Stage', Print=True)
			sys.path.insert(0, self.SIM_PREASTER)

			count, NumActive = 0, 0
			PreError = []
			PreStat = {}
			for Name, StudyDict in self.SimData.items():
				PreAsterFile = StudyDict['Parameters'].PreAsterFile
				if not PreAsterFile: continue
				PreAster = import_module(PreAsterFile)
				PreAsterSgl = getattr(PreAster, 'Single',None)
				if not PreAsterSgl: continue


				self.Logger("'{}' started\n".format(Name),Print=True)
				os.makedirs(StudyDict['PREASTER'],exist_ok=True)

				proc = Process(target=MPRun.main, args=(self,StudyDict,PreAsterSgl))

				if self.mode in ('Interactive','Terminal'):
					proc.start()
				else :
					with open(SimLogFile.format(StudyDict['PREASTER']), 'w') as f:
						with contextlib.redirect_stdout(f):
							# stderr may need to be written to a seperate file and then copied over
							with contextlib.redirect_stderr(sys.stdout):
								proc.start()

				# Copy the parameters file used for this simulation
				shutil.copy("{}/Parameters.py".format(StudyDict['TMP_CALC_DIR']), StudyDict['CALC_DIR'])
				StudyDict['__write__'] = True

				count +=1
				NumActive +=1
				PreStat[Name] = proc
				while NumActive==NumThreads or count==NumSim:
					for tmpName, proc in PreStat.copy().items():
						EC = proc.exitcode
						if EC == None:
							continue
						tmpStudyDict = self.SimData[tmpName]
						if EC == 0:
							self.Logger("'{}' completed\n".format(tmpName),Print=True)
						else :
							self.Logger("'{}' returned error code {}\n".format(tmpName,EC),Print=True)
							PreError.append(tmpName)

						if self.mode in ('Continuous','Headless'):
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

			# If the PreAster file has the function Combind it will be executed here
			PreAster = import_module(SimMaster.PreAsterFile)
			if hasattr(PreAster, 'Combined'):
				self.Logger('Combined function started', Print=True)

				if self.mode in ('Interactive','Terminal'):
					err = PreAster.Combined(self)
				else :
					with open(self.LogFile, 'a') as f:
						with contextlib.redirect_stdout(f):
							# stderr may need to be written to a seperate file and then copied over
							with contextlib.redirect_stderr(sys.stdout):
								err = PreAster.Combined(self)
				if err == None:
					self.Logger('Combined function completed successfully', Print=True)
				else :
					self.Exit("Combined function returned error '{}'".format(err))

				self.Logger('Combined function complete', Print=True)

			self.Logger('>>> PreAster Stage Complete\n', Print=True)

		if RunAster and hasattr(SimMaster,'AsterFile'):
			self.Logger('>>> Aster Stage', Print=True)
			AsterError = []
			AsterStat = {}
			count, NumActive = 0, 0
			for Name, StudyDict in self.SimData.items():
				os.makedirs(StudyDict['ASTER'],exist_ok=True)

				# Create export file for CodeAster
				ExportFile = "{}/Export".format(StudyDict['ASTER'])
				CommFile = '{}/{}.comm'.format(self.SIM_ASTER,StudyDict['Parameters'].AsterFile)
				MessFile = '{}/AsterLog'.format(StudyDict['ASTER'])
				self.CodeAster.ExportWriter(ExportFile, CommFile,
											StudyDict["MeshFile"],
											StudyDict['ASTER'], MessFile)



				self.Logger("Aster for '{}' started".format(Name),Print=True)
				AsterStat[Name] = self.AsterExec(StudyDict, ExportFile)
				count +=1
				NumActive +=1

				# Copy the parameters file used for this simulation, if it's not been written previously
				if not StudyDict.get('__write__'):
					shutil.copy("{}/Parameters.py".format(StudyDict['TMP_CALC_DIR']), StudyDict['CALC_DIR'])
					StudyDict['__write__'] = True

				while NumActive==NumThreads or count==NumSim:
					for tmpName, Proc in AsterStat.copy().items():
						Poll = Proc.poll()
						if Poll == None:
							continue
						tmpStudyDict = self.SimData[tmpName]
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

						if self.mode in ('Continuous','Headless'):
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
			for Name, StudyDict in self.SimData.items():
				PostAsterFile = getattr(StudyDict['Parameters'],'PostAsterFile', None)
				if not PostAsterFile : continue
				PostAster = import_module(PostAsterFile)
				PostAsterSgl = getattr(PostAster, 'Single',None)
				if not PostAsterSgl: continue

				self.Logger("PostAster for '{}' started".format(Name),Print=True)
				if not os.path.isdir(StudyDict['POSTASTER']): os.makedirs(StudyDict['POSTASTER'])

				proc = Process(target=MPRun.main, args=(self,StudyDict,PostAsterSgl))
				if self.mode in ('Interactive','Terminal'):
					proc.start()
				else :
					with open(SimLogFile.format(StudyDict['POSTASTER']), 'w') as f:
						with contextlib.redirect_stdout(f):
							# stderr may need to be written to a seperate file and then copied over
							with contextlib.redirect_stderr(sys.stdout):
								proc.start()

				# Copy the parameters file used for this simulation, if it's not been written previously
				if not StudyDict.get('__write__'):
					shutil.copy("{}/Parameters.py".format(StudyDict['TMP_CALC_DIR']), StudyDict['CALC_DIR'])
					StudyDict['__write__'] = True

				count +=1
				NumActive +=1
				PostStat[Name] = proc
				while NumActive==NumThreads or count==NumSim:
					for tmpName, proc in PostStat.copy().items():
						EC = proc.exitcode
						if EC == None:
							continue

						tmpStudyDict = self.SimData[tmpName]
						if EC == 0:
							self.Logger("Post-Aster for '{}' completed".format(tmpName),Print=True)
						else :
							self.Logger("Post-Aster for '{}' returned error code {}".format(tmpName,EC),Print=True)
							PostError.append(tmpName)

						if self.mode in ('Continuous','Headless'):
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

				if self.mode in ('Interactive','Terminal'):
					err = PostAster.Combined(self)
				else :
					with open(self.LogFile, 'a') as f:
						with contextlib.redirect_stdout(f):
							# stderr may need to be written to a seperate file and then copied over
							with contextlib.redirect_stderr(sys.stdout):
								err = PostAster.Combined(self)

				if err == None:
					self.Logger('Combined function completed successfully', Print=True)
				else :
					self.Exit("Combined function returned error '{}'".format(err))

			self.Logger('>>> PostAster Stage Complete\n', Print=True)

		self.Logger('### Simulations Completed ###',Print=True)

		# Opens up all results in ParaVis
		if ShowRes:
			print("### Opening .rmed files in ParaVis ###\n")
			ResFiles = {}
			for SimName, StudyDict in self.SimData.items():
				for root, dirs, files in os.walk(StudyDict['CALC_DIR']):
					for file in files:
						fname, ext = os.path.splitext(file)
						if ext == '.rmed':
							ResFiles["{}_{}".format(SimName,fname)] = "{}/{}".format(root, file)
			Script = "{}/ShowRes.py".format(self.COM_SCRIPTS)
			SubProc = self.Salome.Run(Script, GUI=True, ArgDict=ResFiles)
			SubProc.wait()


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
		elif self.mode == 'Terminal':
			command = "{} {} ".format(self.ASTER_DIR, exportfile)
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

	def __CheckFile__(self,Directory,fname,ext):
		if not fname:
			return True
		else:
			return os.path.isfile('{}/{}.{}'.format(Directory,fname,ext))


	def __ForceArgs__(self,ArgList):
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
