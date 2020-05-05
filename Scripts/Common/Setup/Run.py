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
	
class VLSetup():
	def __init__(self, Simulation, StudyDir, StudyName, Input, **kwargs):
		'''
		kwargs available:
		port: Give the port number of an open Salome instance to connect to
		mode: 3 options available:
		     - interavtive: All outputs shown in terminal window(s)
		     - continuous: Output written to file throughout execution
		     - headless: Output written to file at the end of execution
		AsterRoot: CodeAster root location. If this is not provided it is assumed it's a part of SalomeMeca
		'''

		port = kwargs.get('port', None)
		mode = kwargs.get('mode', 'headless')
		AsterRoot = kwargs.get('AsterRoot', None)
		ConfigFile = kwargs.get('ConfigFile','VLconfig') 

		# If port is provided it assumes an open instance of salome exists on that port and will shell in to it
		# The second value in the list dictates whether or not to kill the salome instance at the end of the process
		self.__port__ = [port, False]

		# Set running mode	
		self.mode = mode

		# Get AsterRoot from SalomeMeca location if AsterRoot not provided as kwarg
		if AsterRoot: 
			self.ASTER_ROOT = AsterRoot
		else:
			SMDir = os.path.dirname(os.path.dirname(shutil.which("salome")))
			self.ASTER_ROOT = "{}/V2019.0.3_universal/tools/Code_aster_frontend-20190/bin/as_run".format(SMDir)

		VLconfig = __import__(ConfigFile)

		# Get the path to the top level directory, VL_DIR
		frame = inspect.stack()[1]		
		pwd = os.path.dirname(os.path.realpath(frame[0].f_code.co_filename))
		VL_DIR=getattr(VLconfig,"VL_DIR",pwd)
		
		# Initiate variables and run some checks
		### Script directories
		self.SCRIPT_DIR = "{}/Scripts".format(VL_DIR)
		self.COM_SCRIPTS = "{}/Common".format(self.SCRIPT_DIR)
		self.COM_PREPROC = "{}/PreProc".format(self.COM_SCRIPTS)
		self.COM_ASTER = "{}/Aster".format(self.COM_SCRIPTS)
		self.COM_POSTPROC = "{}/PostProc".format(self.COM_SCRIPTS)

		self.SIM_SCRIPTS = "{}/{}".format(self.SCRIPT_DIR, Simulation)
		self.SIM_PREPROC = "{}/PreProc".format(self.SIM_SCRIPTS)
		self.SIM_ASTER = "{}/Aster".format(self.SIM_SCRIPTS)
		self.SIM_POSTPROC = "{}/PostProc".format(self.SIM_SCRIPTS)

		# Materials directory
		self.MATERIAL_DIR = "{}/Materials".format(VL_DIR)

		# Output directories - these are where meshes, Aster results and pre/post-processing will be stored
		OUTPUT_DIR = getattr(VLconfig,'OutputDir',"{}/Output".format(VL_DIR))
		OUTPUT_DIR = OUTPUT_DIR.replace('$VLDir',VL_DIR)
		STUDY_DIR = "{}/{}/{}".format(OUTPUT_DIR, Simulation, StudyDir)
		self.SIM_DIR = "{}/{}".format(STUDY_DIR, StudyName)
		self.MESH_DIR = "{}/Meshes".format(STUDY_DIR)

		### Input dictionary
		self.Input = Input
		self.Input['INPUT_DIR'] = '{}/Input/{}/{}'.format(VL_DIR, Simulation, StudyDir)

		# Create directory in /tmp
		if StudyDir == 'Testing': self.TMP_DIR = '/tmp/test'
		else: self.TMP_DIR = '/tmp/{}_{}'.format(StudyName,(datetime.datetime.now()).strftime("%y%m%d%H%M%S"))

		# Check the Input directory to ensure the the required files exist
		self.ErrorCheck('Input')

	def Create(self, **kwargs):
		'''
		kwargs available:
		RunMesh: Boolean to dictate whether or not to create meshes
		RunAster: Boolean to dictate whether or not to run CodeAster
		'''
		RunMesh = kwargs.get('RunMesh', True)
		RunAster = kwargs.get('RunAster','True')

		sys.path.insert(0, self.Input['INPUT_DIR'])
		Main = __import__(self.Input['Main'])

#		if not os.path.isdir(self.TMP_DIR): os.makedirs(self.TMP_DIR)
		MainDict = copy.deepcopy(self.__dict__)
		MainMesh = getattr(Main, 'Mesh', None)
		MainSim = getattr(Main, 'Sim', None)

		if 'Parametric' in self.Input: Parametric = __import__(self.Input['Parametric'])
		else : Parametric = None

		self.MeshList = []
		self.Studies = {}
		# Create Mesh parameter files if they are required
		if RunMesh and MainMesh:
			if not os.path.isdir(self.MESH_DIR): os.makedirs(self.MESH_DIR)
			self.GEOM_DIR = '{}/Geom'.format(self.TMP_DIR) 
			if not os.path.isdir(self.GEOM_DIR): os.makedirs(self.GEOM_DIR)

			ParaMesh = getattr(Parametric, 'Mesh', None)
			MeshNames = getattr(ParaMesh, 'Name', [MainMesh.Name])

			MeshDict = {MeshName:{} for MeshName in MeshNames}
			for VarName, Value in MainMesh.__dict__.items():
				NewVals = getattr(ParaMesh, VarName, False)
				for i, MeshName in enumerate(MeshNames):
					Val = Value if NewVals==False else NewVals[i]
					MeshDict[MeshName][VarName] = Val

			sys.path.insert(0, self.SIM_PREPROC)
			for MeshName, ParaDict in MeshDict.items():
				self.ErrorCheck('Mesh',MeshDict=ParaDict)
				self.WriteModule("{}/{}.py".format(self.GEOM_DIR, MeshName), ParaDict)
				self.MeshList.append(MeshName)

		# Create Simulation parameter files
		if RunAster and MainSim:
			if not os.path.exists(self.ASTER_ROOT):
				self.Exit("CodeAster location invalid")

			if not os.path.isdir(self.SIM_DIR): os.makedirs(self.SIM_DIR)

			ParaSim = getattr(Parametric, 'Sim', None)
			SimNames = getattr(ParaSim, 'Name', [MainSim.Name])

			SimDict = {SimName:{} for SimName in SimNames}
			for VarName, Value in MainSim.__dict__.items():
				NewVals = getattr(ParaSim, VarName, False)
				for i, SimName in enumerate(SimNames):
					Val = Value if NewVals==False else NewVals[i]
					SimDict[SimName][VarName] = Val

			for SimName, ParaDict in SimDict.items():
				self.ErrorCheck('Simulation',SimDict=ParaDict)
				StudyDict = {}
				# Create simulation related directories
				StudyDict['TMP_CALC_DIR'] = TMP_CALC_DIR = "{}/{}".format(self.TMP_DIR, SimName)
				if not os.path.isdir(TMP_CALC_DIR): os.makedirs(TMP_CALC_DIR)
				StudyDict['CALC_DIR'] = CALC_DIR = "{}/{}".format(self.SIM_DIR, SimName)	
				if not os.path.isdir(CALC_DIR): os.makedirs(CALC_DIR)
				# Can leave these parts until later
				StudyDict['ASTER_DIR'] = ASTER_DIR = "{}/Aster".format(CALC_DIR)
				if not os.path.isdir(ASTER_DIR): os.makedirs(ASTER_DIR)
				StudyDict['OUTPUT_DIR'] = OUTPUT_DIR = "{}/Output".format(CALC_DIR)
				if not os.path.isdir(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

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

	def Mesh(self, **kwargs):
		'''
		kwargs available:
		MeshCheck: input a meshname and it will open this mesh in the GUI
		'''
		MeshCheck = kwargs.get('MeshCheck', None)

		MeshList = getattr(self,'MeshList', None)
		if MeshList:
			print('Starting Meshing\n')
			MeshLog = "{}/Log".format(self.MESH_DIR)
			if self.mode != 'interactive':				
				with open(MeshLog,'w') as f: f.write('Starting Meshing\n')
					
			# Script which is used to import the necessary mesh function
			PreProcScript = '{}/Run.py'.format(self.COM_PREPROC)

			for mesh in self.MeshList:
				MeshStart = "Meshing '{}'".format(mesh)
				print(MeshStart)
				if self.mode != 'interactive': 
					with open(MeshLog,'a') as f: f.write(MeshStart)
				ArgDict = {"Parameters":mesh, "MESH_FILE":"{}/{}.med".format(self.MESH_DIR, mesh)}
				AddPath = [self.SIM_PREPROC, self.GEOM_DIR]
				self.SalomeRun(PreProcScript, AddPath=AddPath, ArgDict=ArgDict, Log = MeshLog)

				MeshFin = "Finished meshing '{}'\n".format(mesh)
				print(MeshFin)
				if self.mode != 'interactive': 
					with open(MeshLog,'a') as f: f.write(MeshFin)

			print('Meshing completed')
			with open(MeshLog,'a') as f: f.write('Meshing completed\n')
		
#		elif self.MeshDict and MeshCheck:
#			if MeshCheck in self.MeshDict:
#				print('### Meshing {} in GUI'.format(MeshCheck))
#				study = self.MeshDict[MeshCheck][0]
#				studydict = self.Studies[study]
#				AddPath = "PYTHONPATH={}:$PYTHONPATH;PYTHONPATH={}:$PYTHONPATH;export PYTHONPATH;".format(self.COM_SCRIPTS,self.SIM_SCRIPTS)				
#				Script = "{}/{}.py".format(self.SIM_PREPROC, studydict['Parameters'].MeshFile)
#				Salome = Popen('{}salome {} args:{}'.format(AddPath,Script,StudyDict['Parameters'].__file__), shell='TRUE')
#				Salome.wait()

#				self.Cleanup()
#				sys.exit()
#			else :
#				self.Exit("Mesh '{}' not listed as one of the meshes which will be ceated".format(str(MeshCheck)))
	
#		# Runs any other pre processing work which must be in the __add file	
#		if os.path.isfile('{}/__add__.py'.format(self.SIM_PREPROC)):
#			from __add__ import Add
#			Add(self)

	def Aster(self, **kwargs):
		'''
		kwargs
		mpi_nbcpu: Num CPUs for parallel CodeAster. Only available if code aster compiled for parallelism.
		mpi_nbnoeud: Num Nodes for parallel CodeAster. Only available if code aster compiled for parallelism.
		ncpus: Number of CPUs for regular CodeAster
		Memory: Amount of memory (Gb) allocated to CodeAster
		'''
		mpi_nbcpu = kwargs.get('mpi_nbcpu',1)
		mpi_nbnoeud = kwargs.get('mpi_nbnoeud',1)
		ncpus = kwargs.get('ncpus',1)
		Memory = kwargs.get('Memory',2)

		if not self.Studies: return

		print('\nStarting Simulations\n')
		SubProcs = {}
		for Name, StudyDict in self.Studies.items():
			AddPath = [self.COM_SCRIPTS, self.TMP_DIR, StudyDict['TMP_CALC_DIR']]
			PythonPath = ["PYTHONPATH={}:$PYTHONPATH;".format(path) for path in AddPath]
			PreCond = PythonPath + ["export PYTHONPATH;export PYTHONDONTWRITEBYTECODE=1;"]
			PreCond = ''.join(PreCond)
			
			# Copy script to tmp folder and add in tmp file location
			commfile = '{0}/{1}.comm'.format(self.SIM_ASTER,StudyDict['Parameters'].CommFile)
			meshfile = "{}/{}.med".format(self.MESH_DIR,StudyDict['Parameters'].Mesh)
			exportfile = "{}/Export".format(StudyDict['ASTER_DIR'])

			# Create export file and write to file
			exportstr = 'P actions make_etude\n' + \
			'P mode batch\n' + \
			'P version stable\n' + \
			'P time_limit 9999\n' + \
			'P mpi_nbcpu {}\n'.format(mpi_nbcpu) + \
			'P mpi_nbnoeud {}\n'.format(mpi_nbnoeud) + \
			'P ncpus {}\n'.format(ncpus) + \
			'P memory_limit {!s}.0\n'.format(1024*Memory) +\
			'F mmed {} D  20\n'.format(meshfile) + \
			'F comm {} D  1\n'.format(commfile) + \
			'F mess {}/AsterLog R  6\n'.format(StudyDict['ASTER_DIR']) + \
			'R repe {} R  0\n'.format(StudyDict['ASTER_DIR'])
			with open(exportfile,'w+') as e:
				e.write(exportstr)

			# Create different command file depending on the mode
			errfile = '{}/Aster.txt'.format(StudyDict['TMP_CALC_DIR'])
			if self.mode == 'interactive':
				xtermset = "-hold -T 'Study: {}' -sb -si -sl 2000".format(Name)
				command = "xterm {} -e '{} {}; echo $? >'{}".format(xtermset, self.ASTER_ROOT, exportfile, errfile)
			elif self.mode == 'continuous':
				command = "{} {} > {}/ContinuousAsterLog ".format(self.ASTER_ROOT, exportfile, StudyDict['ASTER_DIR'])
			else :
				command = "{} {} >/dev/null 2>&1".format(self.ASTER_ROOT, exportfile)

			# Start Aster subprocess
			SubProcs[Name] = Popen(PreCond + command , shell='TRUE')

		# Wait until all Aster subprocesses are finished before moving on
		AsterError = False
		while SubProcs:
			# Check to see the status of each subprocess
			for Name, Proc in SubProcs.copy().items():
				Poll = Proc.poll()
				if Poll is not None:
					err = Poll
					if self.mode == 'interactive':
						with open('{}/Aster.txt'.format(StudyDict['TMP_CALC_DIR']),'r') as f:
							err = int(f.readline())
					elif self.mode == 'continuous':
						os.remove('{}/ContinuousAsterLog'.format(StudyDict['ASTER_DIR']))

					if err != 0:
						print("Error in simulation '{}' - Check the log file".format(Name))
						AsterError = True
					else :
						print("Simulation '{}' completed without errors".format(Name))
					SubProcs.pop(Name)
					Proc.terminate()

			# Check if subprocess has finished every 1 second
			time.sleep(1)

		if AsterError: self.Exit("Some simulations finished with errors")
		print('\nFinished Simulations')

	def PostProc(self, **kwargs):
		'''
		kwargs available:
		ShowRes: Opens up all results files in Salome GUI. Boolean
		'''
		ShowRes = kwargs.get('ShowRes', False)

		if not self.Studies: return

		# Opens up all results in ParaVis
		if ShowRes:
			print("Opening .rmed files in ParaVis")
			ResList=[]
			for study, StudyDict in self.Studies.items():
				ResName = StudyDict['Parameters'].ResName
				if type(ResName) == str: ResName = [ResName]
				elif type(ResName) == dict: ResName=list(ResName.values())
				ResList += ["{0}_{1}={2}/{1}.rmed".format(study,name,StudyDict['ASTER_DIR']) for name in ResName]

			AddPath = "PYTHONPATH={}:$PYTHONPATH;PYTHONPATH={}:$PYTHONPATH;export PYTHONPATH;".format(self.COM_SCRIPTS,self.SIM_SCRIPTS)				
			Script = "{}/ParaVisAll.py".format(self.COM_POSTPROC)
			Salome = Popen('{}salome {} args:{} '.format(AddPath,Script,",".join(ResList)), shell='TRUE')
			Salome.wait()

		sys.path.insert(0, self.SIM_POSTPROC)
		# Run PostCalcFile and ParVis file if they are provided
		for Name, StudyDict in self.Studies.items():
			RunPostProc = getattr(StudyDict['Parameters'],'RunPostProc', 'N')
			if RunPostProc not in ('yes','Yes','y','Y'): continue

			PostCalcFile = getattr(StudyDict['Parameters'],'PostCalcFile', None)
			if PostCalcFile:
				PostP = __import__(PostCalcFile).main
				PostP(self, Name)

			ParaVisFile = getattr(StudyDict['Parameters'],'ParaVisFile', None)
			if ParaVisFile:
				Script = "{}/{}.py".format(self.SIM_POSTPROC, ParaVisFile)
				ArgDict = {"Parameters":StudyDict["Parameters"].__name__, 'StudyDict':self.STUDYDICT}
				self.SalomeRun(Script, AddPath=StudyDict['TMP_CALC_DIR'], ArgDict = ArgDict)

	def ErrorCheck(self, Stage, **kwargs):
		if Stage == 'Input':
			# Check if the Input directory exists
			if not os.path.isdir(self.Input['INPUT_DIR']):
				self.Exit("Directory {0} does not exist\n".format(self.Input['INPUT_DIR']))
			# Check 'Main' is supplied in the dictionary and that the file exists
			Main = self.Input.get('Main', '')
			if Main and not os.path.exists('{}/{}.py'.format(self.Input['INPUT_DIR'], Main)):
				self.Exit("'Main' input file '{}' not in Input directory {}\n".format(self.Input['Main'],self.Input['INPUT_DIR']))
			elif not Main :
				self.Exit("The key 'Main' has not been supplied in the input dictionary")
			# Check that the parametric file exists if it is supplied
			Parametric = self.Input.get('Parametric','')		
			if Parametric and not os.path.exists('{}/{}.py'.format(self.Input['INPUT_DIR'], Parametric)):
				self.Exit("'Parametric' input file '{}' not in Input directory {}\n".format(Parametric,self.Input['INPUT_DIR']))

		if Stage == 'Mesh':
			MeshDict = kwargs.get('MeshDict')
			if os.path.exists('{}/{}.py'.format(self.SIM_PREPROC,MeshDict['File'])):
				MeshFile = __import__(MeshDict['File'])
				ErrorFunc = getattr(MeshFile, 'GeomError', None)
				if ErrorFunc:
					ParamMesh = Namespace()
					ParamMesh.__dict__.update(MeshDict)
					err = ErrorFunc(ParamMesh)
				else : err = None
				if err: self.Exit("GeomError in '{}' - {}".format(MeshDict['Name'], err))
			else:
				self.Exit("Mesh file '{}' does not exist in {}".format(MeshDict['File'], self.SIM_PREPROC))

		if Stage == 'Simulation':
			SimDict = kwargs.get('SimDict')
			# Check that the CommFile exists
			if not os.path.isfile('{}/{}.comm'.format(self.SIM_ASTER,SimDict['CommFile'])):
				self.Exit("CommFile '{}' not in directory '{}'".format(SimDict['CommFile'], self.SIM_ASTER))

			# Check either the mesh is in the mesh directory or that it is in MeshList ready to be created
			if SimDict['Mesh'] in self.MeshList: pass
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
		Log: The log file you want to write information to
		AddPath: Additional paths that Salome will be able to import from
		ArgDict: a dictionary of the arguments that Salome will get
		ArgList: a list of arguments to be passed to Salome
		GUI: Opens a new instance with GUI (useful for testing)
		'''
		Log = kwargs.get('Log', "")
		AddPath = kwargs.get('AddPath',[])
		ArgDict = kwargs.get('ArgDict', {})
		ArgList = kwargs.get('ArgList',[])
		GUI = kwargs.get('GUI',False)

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
			SalomeGUI = Popen(PythonPath + command, shell='TRUE')
			SalomeGUI.wait()
			return


		if not self.__port__[0]:
			portfile = '{}/port.txt'.format(self.TMP_DIR)
			command = 'salome -t --ns-port-log {}'.format(portfile)

			if self.mode != 'interactive':
				command += " >> {} 2>&1".format(Log)

			Salome = Popen(command, shell='TRUE')
			Salome.wait()
			self.CheckProc(Salome,'Salome instance has not been created')
			
			### Get port number from file
			with open(portfile,'r') as f:
				self.__port__ = [int(f.readline()), True]

		command = "salome shell -p{!s} {} args:{}".format(self.__port__[0], Script, Args)
		if self.mode != 'interactive':
			command += " >> {} 2>&1".format(Log)

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




			

