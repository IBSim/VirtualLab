import datetime
import os
import sys
import numpy as np
import shutil
import time
from subprocess import Popen, PIPE, STDOUT
import inspect
import copy
	
class Setup():
	def __init__(self, STUDY_DIR, SIMULATION, STUDY_NAME, INPUT_DICT,**kwargs):
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

		# If port is provided it assumes an open instance of salome exists on that port and will shell in to it
		# The second value in the list dictates whether or not to kill the salome instance at the end of the process
		if port: self.__port__ = [port, False]
		else : self.__port__ = [None, True]

		# Set running mode	
		self.mode = mode

		# Get AsterRoot from SalomeMeca location if AsterRoot not provided as kwarg
		if AsterRoot: 
			self.ASTER_ROOT = AsterRoot
		else:
			SMDir = os.path.dirname(os.path.dirname(shutil.which("salome")))
			self.ASTER_ROOT = "{}/V2019.0.3_universal/tools/Code_aster_frontend-20190/bin/as_run".format(SMDir)

		# Get the path to the top level directory, VL_DIR
		frame = inspect.stack()[1]
		VL_DIR = os.path.dirname(os.path.realpath(frame[0].f_code.co_filename))

		# Initiate variables and run some checks
		### Script directories
		self.SCRIPT_DIR = "{}/Scripts".format(VL_DIR)
		self.COM_SCRIPTS = "{}/Common".format(self.SCRIPT_DIR)
		self.COM_PREPROC = "{}/PreProc".format(self.COM_SCRIPTS)
		self.COM_ASTER = "{}/Aster".format(self.COM_SCRIPTS)
		self.COM_POSTPROC = "{}/PostProc".format(self.COM_SCRIPTS)

		self.SIM_SCRIPTS = "{}/{}".format(self.SCRIPT_DIR, SIMULATION)
		self.SIM_PREPROC = "{}/PreProc".format(self.SIM_SCRIPTS)
		self.SIM_ASTER = "{}/Aster".format(self.SIM_SCRIPTS)
		self.SIM_POSTPROC = "{}/PostProc".format(self.SIM_SCRIPTS)

		# Materials directory
		self.MATERIAL_DIR = "{}/Materials".format(VL_DIR)	

		### Simulation directories
		self.STUDY_DIR = "{}/{}/{}/{}".format(VL_DIR, STUDY_DIR, SIMULATION, STUDY_NAME)
		self.INPUT_DIR = '{}/Input/{}/{}'.format(VL_DIR, SIMULATION, STUDY_NAME)

		# Create directory in /tmp
		if STUDY_NAME == 'Testing': self.TMP_DIR = '/tmp/test'
		else: self.TMP_DIR = '/tmp/{}_{}'.format(STUDY_DIR,(datetime.datetime.now()).strftime("%y%m%d%H%M%S"))

		# Check the Input directory to ensure the the required files exist
		self.ErrorCheck('Input', INPUT_DICT)

	def Create(self):
		# Open main input file
		with open(self.INPUT[0],'r') as g:
			MainScript = g.readlines()

		if self.StudyType == 'Single':
			StudyNames = [self.INPUT[1]]
			if self.INPUT[1] == 'Mesh':
				self.Exit("Name used for Single study is 'Mesh' - this name can't be used")
			# No changes needed to Main file 
			NewScripts = [''.join(MainScript)]

		elif self.StudyType == 'Parametric':
			# Import study names and new values from parametric file
			sys.path.insert(0,self.INPUT_DIR)
			Parametric = __import__(os.path.splitext(os.path.basename(self.INPUT[1]))[0])
			sys.path.pop(0)

			StudyNames = Parametric.CalcName
			numStudy = len(StudyNames)
			if 'Mesh' in StudyNames:
				self.Exit("One of the CalcNames in the parametric file is 'Mesh' - this name can't be used ")

			# Find values which will be changed in Main and extract values from Parametric file 
			inflist = []
			for var in dir(Parametric):
				if var.startswith(("__","CalcName")):
					continue
				newvals = getattr(Parametric, var)
				if len(newvals) < numStudy:
					self.Exit("Fewer entries for variable '{}' than there are studies".format(var))

				inf = [(j, line) for j, line in enumerate(MainScript) if line.startswith(var)]
				linenum, line = inf[0]
				oldval = line.replace('=',' ').split()[1]

				inflist.append((linenum, oldval, newvals))

			# Substitiute parametric values in to a copy of the Main file
			NewScripts = []
			for i, name in enumerate(StudyNames):
				studyscript = MainScript.copy()
				for linenum, oldval, newvals in inflist:
					newval = newvals[i]
					if newval == None:
						continue
					if type(newval) == str:
						newval = "'{}'".format(newval)
					else: 
						newval = str(newval)
					studyscript[linenum] = studyscript[linenum].replace(oldval, newval)

				NewScripts.append(''.join(studyscript))

#		self.GEOM_DIR = self.TMP_DIR + '/Geom' 
#		if not os.path.exists(self.GEOM_DIR):
#			os.makedirs(self.GEOM_DIR)

		self.MESH_DIR = self.STUDY_DIR + '/Mesh'
		if not os.path.exists(self.MESH_DIR):
			os.makedirs(self.MESH_DIR)

		sys.path.insert(0, self.COM_SCRIPTS)
		sys.path.insert(0, self.SIM_PREPROC)
		sys.path.insert(0, self.SIM_SCRIPTS)

		# Gather information for each study
		self.MeshDict = {}
		self.Studies = {}
		MainDict = copy.deepcopy(self.__dict__)
		for name, script in zip(StudyNames, NewScripts):
			# Create nested dictionary for each study
#			self.Studies[name] = {}
			StudyDict = {}
			# Define directories for each study
			StudyDict['CALC_DIR'] = CALC_DIR = "{}/{}".format(self.STUDY_DIR, name)	
			if not os.path.isdir(CALC_DIR):
				os.makedirs(CALC_DIR)
			StudyDict['TMP_CALC_DIR'] = TMP_CALC_DIR = "{}/{}".format(self.TMP_DIR, name)
			if not os.path.isdir(TMP_CALC_DIR):
				os.makedirs(TMP_CALC_DIR)
			StudyDict['ASTER_DIR'] = ASTER_DIR = "{}/Aster".format(CALC_DIR)
			if not os.path.isdir(ASTER_DIR):
				os.makedirs(ASTER_DIR)
			StudyDict['OUTPUT_DIR'] = OUTPUT_DIR = "{}/Output".format(CALC_DIR)
			if not os.path.isdir(OUTPUT_DIR):os.makedirs(OUTPUT_DIR)

			# Write each Parameter script to file and import as module
			ParaName = name
			StudyDict['PARAM_FILE'] = PARAM_FILE = '{}/{}.py'.format(TMP_CALC_DIR,ParaName)
			with open(PARAM_FILE,'w+') as g, open(CALC_DIR + '/Parameters.py','w+') as f:
				g.write(script), f.write(script)
			sys.path.insert(0,TMP_CALC_DIR)
			StudyDict['Parameters'] = Parameters = __import__(ParaName)
			sys.path.pop(0)

			# Define files
			StudyDict['TMP_DICT'] = "{}/tmpDict.py".format(TMP_CALC_DIR)
			StudyDict['TMP_FILE'] = TMP_FILE = TMP_CALC_DIR + '/tmpfile.py'
			StudyDict['MESH_FILE'] = '{}/{}.med'.format(self.MESH_DIR,Parameters.MeshName)
			StudyDict['EXPORT_FILE'] = "{}/Export".format(ASTER_DIR)
#			self.Studies[name]['LOG_FILE'] = "{}/Log".format(CALC_DIR)
#			with open(self.Studies[name]['LOG_FILE'],'w') as f:
#				pass

			# Create tmp file with information in it for salome and code_aster to find
			tmpfileinfo = 'SIM_SCRIPTS = "{}"\n'.format(self.SIM_SCRIPTS) + \
			'MATERIAL_DIR = "{}"\n'.format(self.MATERIAL_DIR) + \
			'PARAM_MOD = "{}"\n'.format(Parameters.__name__) + \
			'CALC_DIR = "{}"\n'.format(StudyDict['CALC_DIR'])
			with open(TMP_FILE, 'w+') as f:
				f.write(tmpfileinfo)

			self.Studies[name] = StudyDict.copy()
			MainDict.update(StudyDict)
			self.DictToFile(StudyDict['TMP_DICT'], WriteDict=MainDict)
			
			# Ensure no mesh generation is duplicated and check for errors in the dimensions
			if Parameters.CreateMesh in ('Yes','yes','Y','y'):
				if Parameters.MeshName not in self.MeshDict:
					self.ErrorCheck('MeshCreate', name)
					self.MeshDict[Parameters.MeshName] = []
				self.MeshDict[Parameters.MeshName].append(name)

			# Run Aster error check and add to run list
#			if Parameters.RunStudy in ('Yes','yes','Y','y'):
#				self.ErrorCheck('Aster',name)

	def DictToFile(self, FileName, **kwargs):
		WriteDict = kwargs.get('WriteDict',{})
		pop = [WriteDict.pop(key) for key, val in WriteDict.copy().items() if inspect.ismodule(val)]
		with open(FileName,'w+') as f:
			f.write("Main={}\n".format(WriteDict))
	def PreProc(self, **kwargs):
		'''
		kwargs available:
		MeshCheck: input a meshname and it will open this mesh in the GUI
		'''
		MeshCheck = kwargs.get('MeshCheck', None)

		if self.MeshDict and not MeshCheck:
			print('Starting Meshing\n')
			MeshLog = "{}/Log".format(self.MESH_DIR)
			if self.mode != 'interactive':				
				with open(MeshLog,'w') as f: f.write('Starting Meshing\n')
					
			# Script which is used to import the necessary mesh function
			PreProcScript = '{}/Run.py'.format(self.COM_PREPROC)

			for meshname, studynames in self.MeshDict.items():
				ParaName = self.Studies[studynames[0]]['Parameters'].__name__
				MeshExport = self.Studies[studynames[0]]['MESH_FILE']

				MeshStart = "Meshing '{}'".format(meshname)
				print(MeshStart)
				if self.mode != 'interactive': 
					with open(MeshLog,'a') as f: f.write(MeshStart)

				ArgDict = {"Parameters":ParaName, "MESH_FILE":MeshExport}
				AddPath = [self.SIM_PREPROC, self.Studies[studynames[0]]['TMP_CALC_DIR']]
				self.SalomeRun(PreProcScript, AddPath=AddPath, ArgDict=ArgDict, Log = MeshLog)

				MeshFin = "'{}' meshed".format(meshname)
				print(MeshFin)
				if self.mode != 'interactive': 
					with open(MeshLog,'a') as f: f.write(MeshFin)

			print('Meshing completed\n')
			with open(MeshLog,'a') as f: f.write('Meshing completed\n')

		elif self.MeshDict and MeshCheck:
			if MeshCheck in self.MeshDict:
				print('### Meshing {} in GUI'.format(MeshCheck))
				study = self.MeshDict[MeshCheck][0]
				studydict = self.Studies[study]
				AddPath = "PYTHONPATH={}:$PYTHONPATH;PYTHONPATH={}:$PYTHONPATH;export PYTHONPATH;".format(self.COM_SCRIPTS,self.SIM_SCRIPTS)				
				Script = "{}/{}.py".format(self.SIM_PREPROC, studydict['Parameters'].MeshFile)
				Salome = Popen('{}salome {} args:{}'.format(AddPath,Script,studydict['PARAM_FILE']), shell='TRUE')
				Salome.wait()

				self.Cleanup()
				sys.exit()
			else :
				self.Exit("Mesh '{}' not listed as one of the meshes which will be ceated".format(str(MeshCheck)))
	
		# Runs any other pre processing work which must be in the __add file	
		if os.path.isfile('{}/__add__.py'.format(self.SIM_PREPROC)):
			from __add__ import Add
			Add(self)

	def Aster(self, **kwargs):
		'''
		kwargs
		mpi_nbcpu: Num CPUs for parallel CodeAster. Only available if code aster compiled for parallelism.
		mpi_nbnoeud: Num Nodes for parallel CodeAster. Only available if code aster compiled for parallelism.
		ncpus: Number of CPUs for regular CodeAster
		Memory: Amount of memory (Gb) allocated to CodeAster
		RunAster: Switch to turn on/off
		'''
		mpi_nbcpu = kwargs.get('mpi_nbcpu',1)
		mpi_nbnoeud = kwargs.get('mpi_nbnoeud',1)
		ncpus = kwargs.get('ncpus',1)
		Memory = kwargs.get('Memory',2)
		RunAster = kwargs.get('RunAster','True')

		if not RunAster:
			return

		print('Starting Simulations')		
		PreCond = 'export PYTHONDONTWRITEBYTECODE=1;export PYTHONPATH="{}";'.format(self.COM_SCRIPTS)
		SubProcs = {}
		for Name, StudyDict in self.Studies.items():
			# Copy script to tmp folder and add in tmp file location
			commfile = '{0}/{1}.comm'.format(self.SIM_ASTER,StudyDict['Parameters'].CommFile)
			tmpcommfile = '{0}/{1}.comm'.format(StudyDict['TMP_CALC_DIR'],StudyDict['Parameters'].CommFile)
			with open(commfile,'r') as g, open(tmpcommfile,'w') as f:
				f.write("TMP_FILE = '{}'\n".format(StudyDict['TMP_FILE']) + g.read())

			# Create export file and write to file
			exportstr = 'P actions make_etude\n' + \
			'P mode batch\n' + \
			'P version stable\n' + \
			'P time_limit 9999\n' + \
			'P mpi_nbcpu {}\n'.format(mpi_nbcpu) + \
			'P mpi_nbnoeud {}\n'.format(mpi_nbnoeud) + \
			'P ncpus {}\n'.format(ncpus) + \
			'P memory_limit {!s}.0\n'.format(1024*Memory) +\
			'F mmed {} D  20\n'.format(StudyDict['MESH_FILE']) + \
			'F comm {} D  1\n'.format(tmpcommfile) + \
			'F mess {}/AsterLog R  6\n'.format(StudyDict['ASTER_DIR']) + \
			'R repe {} R  0\n'.format(StudyDict['ASTER_DIR'])
			with open(StudyDict['EXPORT_FILE'],'w+') as e:
				e.write(exportstr)

			# Create different command file depending on the mode
			errfile = '{}/Aster.txt'.format(StudyDict['TMP_CALC_DIR'])
			if self.mode == 'interactive':
				xtermset = "-hold -T 'Study: {}' -sb -si -sl 2000".format(Name)
				command = "xterm {} -e '{} {}; echo $? >'{}".format(xtermset, self.ASTER_ROOT, StudyDict['EXPORT_FILE'], errfile)
			elif self.mode == 'continuous':
				command = "{} {} > {}/ContinuousAsterLog ".format(self.ASTER_ROOT, StudyDict['EXPORT_FILE'], StudyDict['ASTER_DIR'])
			else :
				command = "{} {} >/dev/null 2>&1".format(self.ASTER_ROOT, StudyDict['EXPORT_FILE'])

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
						with open('{}/Aster.txt'.format(self.Studies[Name]['TMP_CALC_DIR']),'r') as f:
							err = int(f.readline())
					elif self.mode == 'continuous':
						os.remove('{}/ContinuousAsterLog'.format(self.Studies[Name]['ASTER_DIR']))

					if err != 0:
						print("Error in simulation '{}' - Check the log file".format(Name))
						AsterError = True
					else :
						print("Simulation '{}' completed without errors".format(Name))
					SubProcs.pop(Name)
					Proc.terminate()

			# Check if subprocess has finished every 1 second
			time.sleep(1)

		if AsterError:
			self.Exit("Some simulations finished with errors")

		print('Finished Simulations')

	def PostProc(self, **kwargs):
		'''
		kwargs available:
		ShowRes: Opens up all results files in Salome GUI. Boolean
		'''

		ShowRes = kwargs.get('ShowRes', False)

		sys.path.insert(0, self.SIM_POSTPROC)

		# Run PostCalcFile if it is provided
		for Name, StudyDict in self.Studies.items():
			RunPostProc = getattr(StudyDict['Parameters'],'RunPostProc', 'N')
			if RunPostProc not in ('yes','Yes','y','Y'): continue

			PostCalcFile = getattr(StudyDict['Parameters'],'PostCalcFile', None)
			if PostCalcFile:
				PostP = __import__(PostCalcFile).main
				PostP(self, Name)


		# Run ParaVis file if it is provided

		# Opens up all results in ParaVis

		if ShowRes:
			print("Opening .rmed files in ParaVis")
			ResList=[]
			for study, StudyDict in self.Studies.items():
				ResName = StudyDict['Parameters'].ResName
				if type(ResName) == str: ResName = [ResName]
				elif type(ResName) == dict: ResName=list(ResName.values())
				elif type(ResName) == list: pass
				ResList += ["{0}_{1}={2}/{1}.rmed".format(study,name,StudyDict['ASTER_DIR']) for name in ResName]

			AddPath = "PYTHONPATH={}:$PYTHONPATH;PYTHONPATH={}:$PYTHONPATH;export PYTHONPATH;".format(self.COM_SCRIPTS,self.SIM_SCRIPTS)				
			Script = "{}/ParaVisAll.py".format(self.COM_POSTPROC)
			Salome = Popen('{}salome {} args:{} '.format(AddPath,Script,",".join(ResList)), shell='TRUE')
			Salome.wait()



	def ErrorCheck(self, Stage, data = None):
		if Stage == 'Input':
			# Here the data variable is the input dictionary
			# Check if the Input directory exists
			if not os.path.isdir(self.INPUT_DIR):
				self.Exit("Directory {0} not in {1} \n".format(os.path.basename(self.INPUT_DIR),os.path.dirname(self.INPUT_DIR)))

			# Check 'Main' is supplied in the dictionary and that the file exists
			if 'Main' in data.keys():
				Mainfile = '{}/{}.py'.format(self.INPUT_DIR, data['Main'])
				if not os.path.exists(Mainfile):
					self.Exit("The 'Main' file '{}' supplied not found in the Input directory\n".format(data['Main']))
				self.INPUT = [Mainfile]	
			else :
				self.Exit("The key 'Main' has not been supplied in the input dictionary")
							
			if 'Parametric' in data.keys():
				parafile = '{}/{}.py'.format(self.INPUT_DIR, data['Parametric'])
				if not os.path.exists(parafile):
					self.Exit("The 'Parametric' file '{}' supplied not found in the Input directory\n".format(data['Parametric']))
				self.INPUT.append(parafile)
				self.StudyType = 'Parametric'

			elif 'Single' in data.keys():
				self.INPUT.append(data['Single'])
				self.StudyType = 'Single'
			else :
				self.Exit("Neither Parametric or Single supplied as a key in the input directory")

		if Stage == 'MeshCreate':
			# Here the data variable is the name of the studies

			Parameters = self.Studies[data]['Parameters']
			# Check that the pre proc file exists
			if not os.path.exists('{}/{}.py'.format(self.SIM_PREPROC,Parameters.MeshFile)):
				self.Exit("PreProc file '{}' not in directory {}\n".format(Parameters.MeshFile, self.SIM_PREPROC))

			# try to import error module from mesh pre proc file
			try:
				ErrorFunc = __import__(Parameters.MeshFile).error
				err = ErrorFunc(Parameters)
				if err:
					self.Exit('Geometry error -' + err)
			except ModuleNotFoundError :
				print('No error module in Pre Proc file')

		if Stage == 'Aster':
			# Here the data variable is the name of the studies
			# Check that the ASTER_ROOT file exists
			if not os.path.exists("{}".format(self.ASTER_ROOT)):
				self.Exit("CodeAster location invalid")

			Parameters = self.Studies[data]['Parameters']
			# Check either the mesh is in the mesh directory or that it is in MeshDict ready to be created
			Meshfile = self.Studies[data]['MESH_FILE']
			if not os.path.exists(Meshfile) and Parameters.MeshName not in self.MeshDict.keys():
				self.Exit("Mesh '{}' isn't being created and doesn't exist in the mesh directory {}. Is 'CreateMesh' flag incorrect?".format(Parameters.MeshName,os.path.dirname(Meshfile)))

			# Check that the comm file exists
			if not os.path.exists('{}/{}.comm'.format(self.SIM_ASTER,Parameters.CommFile)):
				self.Exit("Comm file '{}' not in directory {}\n".format(Parameters.CommFile, self.SIM_ASTER))

			### Check that materials are in the Material directory
			Materials = getattr(Parameters,'Materials',[])
			if type(Materials)==str: Materials = [Materials]
			elif type(Materials)==dict:Materials = Materials.values()
			for mat in set(Materials):
				if not os.path.exists('{}/{}'.format(self.MATERIAL_DIR, mat)):
					self.Exit("Material {} not in the materials dictionary".format(mat))

	def SalomeRun(self, Script, **kwargs):
		'''
		kwargs available:
		Log: The log file you want to write information to
		AddPath: Additional paths that Salome will be able to import from
		ArgDict: a dictionary of the arguments that Salome will get
		ArgList: a list of arguments to be passed to Salome
		'''
		Log = kwargs.get('Log', "")
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
				self.__port__[0] = int(f.readline())

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




			

