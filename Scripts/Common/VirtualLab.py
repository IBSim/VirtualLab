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

class VLSetup():
	def __init__(self, Simulation, Project, StudyName, Parameters_Master, Parameters_Var, Mode, **kwargs):
		'''
		kwargs available:
		port: Give the port number of an open Salome instance to connect to
		'''
		port = kwargs.get('port', None)

		# If port is provided it assumes an open instance of salome exists on that port
		# and will shell in to it. The second value in the list dictates whether or not
		# to kill the salome instance at the end of the process.
		self.__port__ = [port, False]

		# Set running mode
		if Mode in ('i', 'I', 'interactive', 'Interactive'): self.mode = 'Interactive'
		elif Mode in ('c', 'C', 'continuous', 'Continuous'): self.mode = 'Continuous'
		elif Mode in ('h', 'H', 'headless', 'Headless'): self.mode = 'Headless'
		else : self.Exit("'Mode' is not in 'Interactive','Continuous' or 'Headless'")

		frame = inspect.stack()[1]
		RunFile = os.path.realpath(frame[0].f_code.co_filename)

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
		if Project == '.dev': self.TMP_DIR = "{}/dev".format(TEMP_DIR)
		else: self.TMP_DIR = '{}/{}_{}'.format(TEMP_DIR, Project, (datetime.datetime.now()).strftime("%y%m%d%H%M%S"))

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
		PROJECT_DIR = "{}/{}/{}".format(OUTPUT_DIR, Simulation, Project)
		self.STUDY_DIR = "{}/{}".format(PROJECT_DIR, StudyName)
		self.MESH_DIR = "{}/Meshes".format(PROJECT_DIR)

		# Create dictionary of Parameters info
		self.Parameters = {'Master':Parameters_Master,'Var':Parameters_Var,'Dir':'{}/{}/{}'.format(INPUT_DIR, Simulation, Project)}

	def Create(self, **kwargs):
		'''
		kwargs available:
		RunMesh: Boolean to dictate whether or not to create meshes
		RunSim: Boolean to dictate whether or not to run CodeAster
		'''
		RunMesh = kwargs.get('RunMesh', True)
		RunSim = kwargs.get('RunSim','True')

		sys.path.insert(0, self.COM_SCRIPTS)
		sys.path.insert(0, self.SIM_SCRIPTS)

		# Check the Parameter files exist
		self.ErrorCheck('Parameters')

		sys.path.insert(0, self.Parameters['Dir'])
		Main = __import__(self.Parameters['Master'])
		Var = __import__(self.Parameters['Var']) if self.Parameters['Var'] else None

		MainDict = copy.deepcopy(self.__dict__)
		MainMesh = getattr(Main, 'Mesh', None)
		MainSim = getattr(Main, 'Sim', None)

		self.Parameters_Master = Main

		# Create Mesh parameter files if they are required
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
				if NumVals != NumMeshes: self.Exit("Number of entries for 'Mesh.{}' not equal to number of meshes".format(VarName))

				for i, MeshName in enumerate(MeshNames):
					Val = Value if NewVals==False else NewVals[i]
					MeshDict[MeshName][VarName] = Val

			if hasattr(ParaMesh,'Run'):
				if len(ParaMesh.Run)!=NumMeshes: self.Exit("Number of entries for variable 'Mesh.Run' not equal to number of meshes")
				MeshNames = [mesh for mesh, flag in zip(MeshNames, ParaMesh.Run) if flag]

			sys.path.insert(0, self.SIM_MESH)
			self.MeshList = []
			for MeshName in MeshNames:
				ParaDict=MeshDict[MeshName]
				self.ErrorCheck('Mesh',MeshDict=ParaDict)
				self.WriteModule("{}/{}.py".format(self.GEOM_DIR, MeshName), ParaDict)
				self.MeshList.append(MeshName)

		# Create Simulation parameter files
		if RunSim and MainSim:
			if not os.path.exists(self.ASTER_DIR):
				self.Exit("CodeAster location invalid")

			if not os.path.isdir(self.STUDY_DIR): os.makedirs(self.STUDY_DIR)

			ParaSim = getattr(Var, 'Sim', None)
			SimNames = getattr(ParaSim, 'Name', [MainSim.Name])
			NumSims = len(SimNames)

			SimDict = {SimName:{} for SimName in SimNames}
			for VarName, Value in MainSim.__dict__.items():
				NewVals = getattr(ParaSim, VarName, False)
				# Check the number of NewVals is correct
				NumVals = len(NewVals) if NewVals else NumSims
				if NumVals!=NumSims: self.Exit("Number of entries for 'Sim.{}' not equal to number of simulations".format(VarName))

				for i, SimName in enumerate(SimNames):
					Val = Value if NewVals==False else NewVals[i]
					SimDict[SimName][VarName] = Val

			if hasattr(ParaSim,'Run'):
				if len(ParaSim.Run)!=NumSims: self.Exit("Number of entries for variable 'Sim.Run' not equal to number of simulations")
				SimNames = [sim for sim, flag in zip(SimNames, ParaSim.Run) if flag]

			self.Studies = {}
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

	def Mesh(self, **kwargs):
		if not hasattr(self, 'MeshList'): return
		'''
		kwargs available:
		MeshCheck: input a meshname and it will open this mesh in the GUI
		ShowMesh: Opens up the meshes in the GUI. Boolean
		'''
		MeshCheck = kwargs.get('MeshCheck', None)
		ShowMesh = kwargs.get('ShowMesh', False)

		MeshList = getattr(self,'MeshList', None)

		# MeshCheck routine which allows you to mesh in the GUI (Used for debugging).
		# The script will terminate after this routine
		if MeshCheck and MeshCheck in MeshList:
			print('### Meshing {} in GUI ###\n'.format(MeshCheck))
			sys.path.insert(0, self.GEOM_DIR)
			MeshParameters = __import__(MeshCheck)

			AddPath = "PYTHONPATH={}:{}:$PYTHONPATH;export PYTHONPATH;".format(self.COM_SCRIPTS,self.SIM_SCRIPTS)
			Script = "{}/{}.py".format(self.SIM_MESH, MeshParameters.File)
			Salome = Popen('{}salome {} args:{}'.format(AddPath,Script,MeshParameters.__file__), shell='TRUE')
			Salome.wait()

			self.Cleanup()
			sys.exit()
		elif MeshCheck and MeshCheck not in MeshList:
			self.Exit("MeshCheck '{}' is not in the list of meshes to be created. Meshes to be created are {}".format(MeshCheck, MeshList))

		print('### Starting Meshing ###\n')
		# This will start a salome instance if one hasnt been proivded with the kwarg 'port' on Setup
		MeshLog = "{}/Log".format(self.MESH_DIR)
		self.SalomeRun(None, SalomeInit=True, OutLog=MeshLog)

		# Script which is used to import the necessary mesh function
		MeshScript = '{}/MeshRun.py'.format(self.COM_SCRIPTS)
		for mesh in self.MeshList:
			print("Starting mesh '{}'".format(mesh))

			IndMeshLog = "{}/Log".format(self.GEOM_DIR)
			ArgDict = {"Parameters":mesh, "MESH_FILE":"{}/{}.med".format(self.MESH_DIR, mesh)}
			AddPath = [self.SIM_MESH, self.GEOM_DIR]
			self.SalomeRun(MeshScript, AddPath=AddPath, ArgDict=ArgDict, OutLog=IndMeshLog)

			IndMeshData = "{}/{}.py".format(self.MESH_DIR, mesh)
			with open(IndMeshData,"w") as g:
				with open("{}/{}.py".format(self.GEOM_DIR,mesh),'r') as MeshData:
					g.write("# Geom & Mesh Parameters\n" + MeshData.read())
				if self.mode != 'Interactive':
					with open(IndMeshLog,'r') as rIndMeshLog:
						g.write("\n'''\n# Meshing log\n{}\n'''".format(rIndMeshLog.read()))

			if self.mode == 'Interactive': print("Completed mesh '{}'\n".format(mesh))
			else : print("Completed mesh '{}'. See '{}' for log\n".format(mesh,IndMeshData))

		print('### Meshing Completed ###\n')
		if ShowMesh:
			print("Opening mesh files in Salome")
			MeshPaths = ["{}/{}.med".format(self.MESH_DIR, name) for name in self.MeshList]
			Salome = Popen('salome {}/ShowMesh.py args:{} '.format(self.COM_SCRIPTS,",".join(MeshPaths)), shell='TRUE')
			Salome.wait()

			self.Cleanup()
			sys.exit()

	def Sim(self, **kwargs):
		if not hasattr(self,'Studies'): return

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

		print('\n### Starting Simulations ###\n')

		SimMaster = self.Parameters_Master.Sim
		if RunPreAster and hasattr(SimMaster,'PreAsterFile'):
			sys.path.insert(0, self.SIM_PREASTER)
			for Name, StudyDict in self.Studies.items():
				PreAsterFile = getattr(StudyDict['Parameters'],'PreAsterFile', None)
				# if not hasattr(StudyDict['Parameters'],'PreSimFile'): continue
				print("Pre-Aster for '{}' started".format(Name))
				if not os.path.isdir(StudyDict['PREASTER']): os.makedirs(StudyDict['PREASTER'])
				PreAster = import_module(PreAsterFile)
				PreAster.main(self, StudyDict)
				print("Pre-Aster for '{}' completed".format(Name))

		if RunAster and hasattr(SimMaster,'AsterFile'):
			mpi_nbcpu = kwargs.get('mpi_nbcpu',1)
			mpi_nbnoeud = kwargs.get('mpi_nbnoeud',1)
			ncpus = kwargs.get('ncpus',1)
			memory = kwargs.get('memory',2)

			SubProcs = {}
			for Name, StudyDict in self.Studies.items():
				if not os.path.isdir(StudyDict['ASTER']): os.makedirs(StudyDict['ASTER'])

				AddPath = [self.COM_SCRIPTS, self.TMP_DIR, StudyDict['TMP_CALC_DIR']]
				PythonPath = ["PYTHONPATH={}:$PYTHONPATH;".format(path) for path in AddPath]
				PreCond = PythonPath + ["export PYTHONPATH;export PYTHONDONTWRITEBYTECODE=1;"]
				PreCond = ''.join(PreCond)

				# Copy script to tmp folder and add in tmp file location
				asterfile = '{0}/{1}.comm'.format(self.SIM_ASTER,StudyDict['Parameters'].AsterFile)
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

				# Create different command file depending on the mode
				errfile = '{}/Aster.txt'.format(StudyDict['TMP_CALC_DIR'])
				if self.mode == 'Interactive':
					xtermset = "-hold -T 'Study: {}' -sb -si -sl 2000".format(Name)
					command = "xterm {} -e '{} {}; echo $? >'{}".format(xtermset, self.ASTER_DIR, exportfile, errfile)
				elif self.mode == 'Continuous':
					command = "{} {} > {}/ContinuousAsterLog ".format(self.ASTER_DIR, exportfile, StudyDict['ASTER'])
				else :
					command = "{} {} >/dev/null 2>&1".format(self.ASTER_DIR, exportfile)

				# Start Aster subprocess
				SubProcs[Name] = Popen(PreCond + command , shell='TRUE')
				print("Aster for '{}' started".format(Name))

			# Wait until all Aster subprocesses are finished before moving on
			AsterError = False
			while SubProcs:
				# Check to see the status of each subprocess
				for Name, Proc in SubProcs.copy().items():
					Poll = Proc.poll()
					if Poll is not None:

						err = Poll
						if self.mode == 'Interactive':
							with open('{}/Aster.txt'.format(self.Studies[Name]['TMP_CALC_DIR']),'r') as f:
								err = int(f.readline())
						elif self.mode == 'Continuous':
							os.remove('{}/ContinuousAsterLog'.format(self.Studies[Name]['ASTER']))

						if err != 0:
							print("Error in simulation '{}' - Check the log file".format(Name))
							AsterError = True
						else :
							print("Aster for '{}' completed".format(Name))
						SubProcs.pop(Name)
						Proc.terminate()

				# Check if subprocess has finished every 1 second
				time.sleep(1)

			if AsterError: self.Exit("Some simulations finished with errors")

		if RunPostAster and hasattr(SimMaster,'PostAsterFile'):
			sys.path.insert(0, self.SIM_POSTASTER)
			for Name, StudyDict in self.Studies.items():
				PostAsterFile = getattr(StudyDict['Parameters'],'PostAsterFile', None)
				if not PostAsterFile : continue
				PostAster = __import__(PostAsterFile)
				if hasattr(PostAster, 'Individual'):
					print("PostAster for '{}' started\n".format(Name))
					if not os.path.isdir(StudyDict['POSTASTER']): os.makedirs(StudyDict['POSTASTER'])
					if self.mode == 'Interactive':
						PostAster.Individual(self, StudyDict)
					else:
						with open("{}/log.txt".format(StudyDict['POSTASTER']), 'w') as f:
							with contextlib.redirect_stdout(f):
								PostAster.Individual(self, StudyDict)
					print("PostAster for '{}' completed\n".format(Name))

			PostAster = import_module(SimMaster.PostAsterFile)
			if hasattr(PostAster, 'Combined'):
				PostAster.Combined(self)
			

		print('\n### Simulations Completed ###')

		# Opens up all results in ParaVis
		if kwargs.get('ShowRes', False):
			print("### Opening .rmed files in ParaVis ###\n")
			ResFiles = {}
			for SimName, StudyDict in self.Studies.items():
				for root, dirs, files in os.walk(StudyDict['CALC_DIR']):
					for file in files:
						fname, ext = os.path.splitext(file)
						if ext == '.rmed':
							ResFiles["{}_{}".format(SimName,fname)] = "{}/{}".format(root, file)
			Script = "{}/ShowRes.py".format(self.COM_SCRIPTS)
			self.SalomeRun(Script, GUI=True, AddPath=[self.COM_SCRIPTS], ArgDict=ResFiles)

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

			# Check either the mesh is in the mesh directory or that it is in MeshList ready to be created
			if SimDict['Mesh'] in getattr(self, 'MeshList', []): pass
			elif os.path.isfile("{}/{}.med".format(self.MESH_DIR, SimDict['Mesh'])): pass
			else : self.Exit("Mesh '{}' isn't being created and is not in the mesh directory '{}'".format(SimDict['Mesh'], self.MESH_DIR))

			Materials = SimDict.get('Materials',[])
			if type(Materials)==str: Materials = [Materials]
			elif type(Materials)==dict:Materials = Materials.values()
			for mat in set(Materials):
				if not os.path.exists('{}/{}'.format(self.MATERIAL_DIR, mat)):
					self.Exit("Material '{}' isn't in the materials directory '{}'".format(mat, self.MATERIAL_DIR))


	def SalomeRun(self, Script, **kwargs):
		'''
		kwargs available:
		OutLog: The log file you want to write stdout to (default is /dev/null)
		ErrLog: The log file you want to write stderr to (default is OutLog)
		AddPath: Additional paths that Salome will be able to import from
		ArgDict: a dictionary of the arguments that Salome will get
		ArgList: a list of arguments to be passed to Salome
		GUI: Opens a new instance with GUI (useful for testing)
		SalomeInit: Creates a new Salome instance in terminal mode
		'''
		OutLog = kwargs.get('OutLog', "/dev/null")
		ErrLog = kwargs.get('ErrLog', OutLog)
		AddPath = kwargs.get('AddPath',[])
		ArgDict = kwargs.get('ArgDict', {})
		ArgList = kwargs.get('ArgList',[])
		GUI = kwargs.get('GUI',False)
		SalomeInit = kwargs.get('SalomeInit',False)

		# Add paths provided to python path for subprocess (self.COM_SCRIPTS and self.SIM_SCRIPTS is always added to path)
		AddPath = [AddPath] if type(AddPath) == str else AddPath
		AddPath += [self.COM_SCRIPTS, self.SIM_SCRIPTS]
		PythonPath = ["PYTHONPATH={}:$PYTHONPATH;".format(path) for path in AddPath]
		PythonPath.append("export PYTHONPATH;")
		PythonPath = "".join(PythonPath)

		# Write ArgDict and ArgList in format to pass to salome
		Args = ["{}={}".format(key, value) for key, value in ArgDict.items()]
		Args = ",".join(ArgList + Args)

		if GUI:
			command = "salome {} args:{}".format(Script, Args)
			GUI = Popen(PythonPath + command, shell='TRUE')
			GUI.wait()
			return


		if not self.__port__[0]:
			# Cd to TMP_DIR to avoid test.out file created in VL
			portfile = '{}/port.txt'.format(self.TMP_DIR)
			command = 'cd {};salome -t --ns-port-log {}'.format(self.TMP_DIR,portfile)

			if self.mode != 'Interactive':
				command += " > {} 2>&1".format(OutLog)

			Salome = Popen(command, shell='TRUE')
			Salome.wait()
			self.CheckProc(Salome,'Salome instance has not been created')
			print("")

			### Get port number from file
			with open(portfile,'r') as f:
				self.__port__ = [int(f.readline()), True]

		# Return here if SalomeInit as we only want to initiate Salome, not run anything
		if SalomeInit: return

		command = "salome shell -p{!s} {} args:{}".format(self.__port__[0], Script, Args)
		if self.mode != 'Interactive':
			command += " 2>{} 1>{}".format(ErrLog, OutLog)

		Salome = Popen(PythonPath + command, shell='TRUE')
		Salome.wait()

		self.CheckProc(Salome)

	def CheckProc(self,Proc,message=''):
		if Proc.returncode != 0:
			self.Exit('Error in subprocess:' + message)

	def Exit(self,Error):
		self.Cleanup('n')
		sys.exit('Error: ' + Error)

	def Cleanup(self,remove = 'y'):
		# If a port is a kwarg during setup it wont be killed, otherwise the instance set up will be killed
		if self.__port__[1]:
			Salome_close = Popen('salome kill {}'.format(self.__port__[0]), shell = 'TRUE')
			Salome_close.wait()

		if remove == 'y':
			shutil.rmtree(self.TMP_DIR)
