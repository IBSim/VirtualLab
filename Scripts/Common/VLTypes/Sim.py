
import os
import sys
sys.dont_write_bytecode=True
from types import SimpleNamespace as Namespace
from importlib import import_module
from contextlib import redirect_stderr, redirect_stdout
from multiprocessing import Process
from pathos.multiprocessing import ProcessPool
import copy

# from Scripts.Common.VLPackages import CodeAster
from ..VLPackages import CodeAster

def CheckFile(Directory,fname,ext):
    if not fname:
        return True
    else:
        return os.path.isfile('{}/{}.{}'.format(Directory,fname,ext))

def Setup(VL,**kwargs):

    os.makedirs(VL.STUDY_DIR, exist_ok=True)
    VL.SimData = {}

    if not kwargs.get('RunSim', True): return

    MetaInfo = {key:val for key,val in VL.__dict__.items() if type(val)==str}

    SimDicts = VL.CreateParameters(VL.Parameters_Master, VL.Parameters_Var,'Sim')
    for SimName, ParaDict in SimDicts.items():
        # Run checks
        # Check files exist
        if not CheckFile(VL.SIM_PREASTER,ParaDict.get('PreAsterFile'),'py'):
        	VL.Exit("PreAsterFile '{}.py' not in directory {}".format(ParaDict['PreAsterFile'],VL.SIM_PREASTER))
        if not CheckFile(VL.SIM_ASTER,ParaDict.get('AsterFile'),'comm'):
        	VL.Exit("AsterFile '{}.comm' not in directory {}".format(ParaDict['AsterFile'],VL.SIM_ASTER,))
        if not CheckFile(VL.SIM_POSTASTER, ParaDict.get('PostAsterFile'), 'py'):
        	VL.Exit("PostAsterFile '{}.py' not in directory {}".format(ParaDict['PostAsterFile'],VL.SIM_POSTASTER))
        # Check mesh will be available
        if not (ParaDict['Mesh'] in VL.MeshData or CheckFile(VL.MESH_DIR, ParaDict['Mesh'], 'med')):
        	VL.Exit("Mesh '{}' isn't being created and is not in the mesh directory '{}'".format(ParaDict['Mesh'], VL.MESH_DIR))
        # Check materials used
        Materials = ParaDict.get('Materials',[])
        if type(Materials)==str: Materials = [Materials]
        elif type(Materials)==dict: Materials = Materials.values()
        MatErr = [mat for mat in set(Materials) if not os.path.isdir('{}/{}'.format(VL.MATERIAL_DIR, mat))]
        if MatErr:
        		VL.Exit("Material(s) {} specified for {} not available.\n"\
        		"Please see the materials directory {} for options.".format(MatErr,SimName,VL.MATERIAL_DIR))
        # Checks complete

        # Create dict of simulation specific information to be nested in SimData
        StudyDict = {}
        StudyDict['TMP_CALC_DIR'] = TMP_CALC_DIR = "{}/{}".format(VL.TMP_DIR, SimName)
        StudyDict['CALC_DIR'] = CALC_DIR = "{}/{}".format(VL.STUDY_DIR, SimName)
        StudyDict['PREASTER'] = "{}/PreAster".format(CALC_DIR)
        StudyDict['ASTER'] = "{}/Aster".format(CALC_DIR)
        StudyDict['POSTASTER'] = "{}/PostAster".format(CALC_DIR)
        StudyDict['MeshFile'] = "{}/{}.med".format(VL.MESH_DIR, ParaDict['Mesh'])

        # Create tmp directory & add __init__ file so that it can be treated as a package
        if not os.path.isdir(TMP_CALC_DIR): os.makedirs(TMP_CALC_DIR)
        with open("{}/__init__.py".format(TMP_CALC_DIR),'w') as f: pass
        # Combine Meta information with that from Study dict and write to file for salome/CodeAster to import
        VL.WriteModule("{}/PathVL.py".format(TMP_CALC_DIR), {**MetaInfo, **StudyDict})
        # Write Sim Parameters to file for Salome/CodeAster to import
        VL.WriteModule("{}/Parameters.py".format(TMP_CALC_DIR), ParaDict)

        # Attach Parameters to StudyDict for ease of access
        StudyDict['Parameters'] = Namespace(**ParaDict)
        # Add StudyDict to SimData dictionary
        VL.SimData[SimName] = StudyDict.copy()

def PoolRun(VL, StudyDict, kwargs):
    RunPreAster = kwargs.get('RunPreAster',True)
    RunAster = kwargs.get('RunAster', True)
    RunPostAster = kwargs.get('RunPostAster', True)

    Parameters = StudyDict["Parameters"]

    if VL.mode == 'Headless':OutFile = "{}/Output.log".format(StudyDict['TMP_CALC_DIR'])
    elif VL.mode == 'Continuous':OutFile = "{}/Output.log".format(StudyDict['TMP_CALC_DIR'])
    else : OutFile=''

    # Copy original dictionary so that we can tell if anything gets added to it
    OrigDict = copy.deepcopy(StudyDict)
    Error=None

    # Create a blank file
    if OutFile:
        with open(OutFile,'w') as f: pass

    if RunPreAster and hasattr(Parameters,'PreAsterFile'):
        sys.path.insert(0, VL.SIM_PREASTER)
        PreAster = import_module(Parameters.PreAsterFile)
        PreAsterSgl = getattr(PreAster, 'Single',None)

        VL.Logger("Running PreAster for '{}'\n".format(Parameters.Name),Print=True)
        os.makedirs(StudyDict['PREASTER'],exist_ok=True)

        if not OutFile:
            err = PreAsterSgl(VL,StudyDict)
        else :
            with open(OutFile,'a') as f:
                with redirect_stdout(f), redirect_stderr(f):
                    err = PreAsterSgl(VL,StudyDict)

        if err:
            Error = ['PreAster',err]
            RunAster=RunPostAster=False

    if RunAster and hasattr(Parameters,'AsterFile'):
        VL.Logger("Running Aster for '{}'\n".format(Parameters.Name),Print=True)

        os.makedirs(StudyDict['ASTER'],exist_ok=True)
        # Create export file for CodeAster
        ExportFile = "{}/Export".format(StudyDict['ASTER'])
        CommFile = '{}/{}.comm'.format(VL.SIM_ASTER, Parameters.AsterFile)
        MessFile = '{}/AsterLog'.format(StudyDict['ASTER'])
        VL.CodeAster.ExportWriter(ExportFile, CommFile,
        							StudyDict["MeshFile"],
        							StudyDict['ASTER'],
                                    MessFile, **kwargs)

        if VL.mode == 'Headless': AsterOut='/dev/null'
        elif VL.mode == 'Continuous': AsterOut = OutFile.format(StudyDict['ASTER'])
        else : AsterOut=''

        SubProc = VL.CodeAster.Run(ExportFile, Name=Parameters.Name,
                                     AddPath=[VL.TMP_DIR,StudyDict['TMP_CALC_DIR']],
                                     OutFile=AsterOut)
        err = SubProc.wait()
        if err:
            Error = ['Aster',"Aster SubProc returned code {}".format(err)]
            RunPostAster=False

        # from subprocess import Popen
        # SubProc = Popen(['echo','Hello World'])
        # SubProc.wait()

    if RunPostAster and hasattr(Parameters,'PostAsterFile'):
        sys.path.insert(0, VL.SIM_POSTASTER)
        PostAster = import_module(Parameters.PostAsterFile)
        PostAsterSgl = getattr(PostAster, 'Single', None)

        VL.Logger("Running PostAster for '{}'\n".format(Parameters.Name),Print=True)
        os.makedirs(StudyDict['POSTASTER'],exist_ok=True)

        if not OutFile:
            err = PostAsterSgl(VL,StudyDict)
        else :
            with open(OutFile,'a') as f:
                with redirect_stdout(f), redirect_stderr(f):
                    err = PostAsterSgl(VL,StudyDict)
        if err:
            Error = ['PostAster',err]

    # Data structure which can be easily returned holdign additional information
    Returner = Namespace(Error=Error)
    # Add StudyDict to returned if it's changed
    if not OrigDict == StudyDict:
        Returner.StudyDict = StudyDict

    return Returner

def devRun(VL,**kwargs):
    if not VL.SimData: return
    kwargs.update(VL.GetArgParser()) # Update with any kwarg passed in the call

    ShowRes = kwargs.get('ShowRes', False)
    NumThreads = kwargs.get('NumThreads',1)

    # Run high throughput part in parallel
    NumSim = len(VL.SimData)
    NumThreads = min(NumThreads,NumSim)

    Arg0 = [VL]*NumSim
    Arg1 = list(VL.SimData.values())
    Arg2 = [kwargs]*NumSim
    if 1:
        pool = ProcessPool(nodes=NumThreads)
        Res = pool.map(PoolRun, Arg0, Arg1, Arg2)
    else :
        from pyina.launchers import MpiPool
        pool = MpiPool(nodes=NumThreads)
        Res = pool.map(PoolRun, Arg0, Arg1, Arg2)

    SimError = []
    for Name, Returner in zip(VL.SimData.keys(),Res):
        if Returner.Error:
            SimError.append(Name)
            VL.Logger("'{}' finished with errors".format(Name),Print=True)
        else :
            VL.Logger("'{}' completed successfully".format(Name), Print=True)
        if hasattr(Returner,'StudyDict'):
            VL.SimData[Name].update(Returner.StudyDict)

    if SimError:
        VL.Exit("The following Simulation routine(s) finished with errors:\n{}".format(SimError))

    PostAster = getattr(VL.Parameters_Master.Sim, 'PostAsterFile', None)
    if PostAster:
        sys.path.insert(0, VL.SIM_POSTASTER)
        PostAster = import_module(PostAster)
        if hasattr(PostAster, 'Combined'):
            VL.Logger('Combined function started', Print=True)
            if VL.mode in ('Interactive','Terminal'):
                err = PostAster.Combined(VL)
            else :
                with open(VL.LogFile, 'a') as f:
                    with redirect_stdout(f), redirect_stderr(f):
                        err = PostAster.Combined(VL)

            if err == None:
                VL.Logger('Combined function completed successfully', Print=True)
            else :
                VL.Exit("Combined function returned error '{}'".format(err))

def Run(VL, **kwargs):
    if not VL.SimData: return
    kwargs.update(VL.GetArgParser()) # Update with any kwarg passed in the call
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

    VL.Logger('\n### Starting Simulations ###\n', Print=True)

    NumSim = len(VL.SimData)
    NumThreads = min(NumThreads,NumSim)

    SimLogFile = "{}/Output.log"

    SimMaster = VL.Parameters_Master.Sim
    if RunPreAster and hasattr(SimMaster,'PreAsterFile'):
    	VL.Logger('>>> PreAster Stage', Print=True)
    	sys.path.insert(0, VL.SIM_PREASTER)

    	count, NumActive = 0, 0
    	PreError = []
    	PreStat = {}
    	for Name, StudyDict in VL.SimData.items():
    		PreAsterFile = StudyDict['Parameters'].PreAsterFile
    		if not PreAsterFile: continue
    		PreAster = import_module(PreAsterFile)
    		PreAsterSgl = getattr(PreAster, 'Single',None)
    		if not PreAsterSgl: continue


    		VL.Logger("'{}' started\n".format(Name),Print=True)
    		os.makedirs(StudyDict['PREASTER'],exist_ok=True)

    		proc = Process(target=MPRun.main, args=(VL,StudyDict,PreAsterSgl))

    		if VL.mode in ('Interactive','Terminal'):
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
    				tmpStudyDict = VL.SimData[tmpName]
    				if EC == 0:
    					VL.Logger("'{}' completed\n".format(tmpName),Print=True)
    				else :
    					VL.Logger("'{}' returned error code {}\n".format(tmpName,EC),Print=True)
    					PreError.append(tmpName)

    				if VL.mode in ('Continuous','Headless'):
    					VL.Logger("See {} for details".format(SimLogFile.format(tmpStudyDict['PREASTER'])),Print=EC)

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

    	if PreError: VL.Exit("The following PreAster routine(s) finished with errors:\n{}".format(PreError),KeepDirs=PreError)

    	# If the PreAster file has the function Combind it will be executed here
    	PreAster = import_module(SimMaster.PreAsterFile)
    	if hasattr(PreAster, 'Combined'):
    		VL.Logger('Combined function started', Print=True)

    		if VL.mode in ('Interactive','Terminal'):
    			err = PreAster.Combined(VL)
    		else :
    			with open(VL.LogFile, 'a') as f:
    				with contextlib.redirect_stdout(f):
    					# stderr may need to be written to a seperate file and then copied over
    					with contextlib.redirect_stderr(sys.stdout):
    						err = PreAster.Combined(VL)
    		if err == None:
    			VL.Logger('Combined function completed successfully', Print=True)
    		else :
    			VL.Exit("Combined function returned error '{}'".format(err))

    		VL.Logger('Combined function complete', Print=True)

    	VL.Logger('>>> PreAster Stage Complete\n', Print=True)

    if RunAster and hasattr(SimMaster,'AsterFile'):
    	VL.Logger('>>> Aster Stage', Print=True)
    	AsterError = []
    	AsterStat = {}
    	count, NumActive = 0, 0

    	for Name, StudyDict in VL.SimData.items():
    		os.makedirs(StudyDict['ASTER'],exist_ok=True)

    		# Create export file for CodeAster
    		ExportFile = "{}/Export".format(StudyDict['ASTER'])
    		CommFile = '{}/{}.comm'.format(VL.SIM_ASTER,StudyDict['Parameters'].AsterFile)
    		MessFile = '{}/AsterLog'.format(StudyDict['ASTER'])
    		VL.CodeAster.ExportWriter(ExportFile, CommFile,
    									StudyDict["MeshFile"],
    									StudyDict['ASTER'], MessFile)

    		VL.Logger("Aster for '{}' started".format(Name),Print=True)

    		if VL.mode == 'Headless': Outfile='/dev/null'
    		elif VL.mode == 'Continuous': Outfile=SimLogFile.format(StudyDict['ASTER'])
    		else : Outfile=''

    		AsterStat[Name] = VL.CodeAster.Run(ExportFile, Name=Name, OutFile=Outfile, AddPath=[VL.TMP_DIR,StudyDict['TMP_CALC_DIR']])

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

    				tmpStudyDict = VL.SimData[tmpName]
    				if Poll == 0:
    					VL.Logger("Aster for '{}' completed".format(tmpName),Print=True)
    				else :
    					VL.Logger("Aster for '{}' returned error code {}.".format(tmpName,Poll))
    					AsterError.append(tmpName)

    				if VL.mode in ('Continuous','Headless'):
    					VL.Logger("See {}/AsterLog for details".format(tmpStudyDict['ASTER']),Print=Poll)

    				AsterStat.pop(tmpName)
    				Proc.terminate()

    			if not len(AsterStat): break
    			time.sleep(0.1)

    	if AsterError: VL.Exit("The following simulation(s) finished with errors:\n{}".format(AsterError),KeepDirs=AsterError)

    	VL.Logger('>>> Aster Stage Complete\n', Print=True)

    if RunPostAster and hasattr(SimMaster,'PostAsterFile'):
    	sys.path.insert(0, VL.SIM_POSTASTER)
    	VL.Logger('>>> PostAster Stage', Print=True)

    	count, NumActive = 0, 0
    	PostError = []
    	PostStat = {}
    	for Name, StudyDict in VL.SimData.items():
    		PostAsterFile = getattr(StudyDict['Parameters'],'PostAsterFile', None)
    		if not PostAsterFile : continue
    		PostAster = import_module(PostAsterFile)
    		PostAsterSgl = getattr(PostAster, 'Single',None)
    		if not PostAsterSgl: continue

    		VL.Logger("PostAster for '{}' started".format(Name),Print=True)
    		if not os.path.isdir(StudyDict['POSTASTER']): os.makedirs(StudyDict['POSTASTER'])

    		proc = Process(target=MPRun.main, args=(VL,StudyDict,PostAsterSgl))
    		if VL.mode in ('Interactive','Terminal'):
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

    				tmpStudyDict = VL.SimData[tmpName]
    				if EC == 0:
    					VL.Logger("Post-Aster for '{}' completed".format(tmpName),Print=True)
    				else :
    					VL.Logger("Post-Aster for '{}' returned error code {}".format(tmpName,EC),Print=True)
    					PostError.append(tmpName)

    				if VL.mode in ('Continuous','Headless'):
    					VL.Logger("See {} for details".format(SimLogFile.format(tmpStudyDict['POSTASTER'])),Print=EC)

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

    	if PostError: VL.Exit("The following PostAster routine(s) finished with errors:\n{}".format(PostError), KeepDirs=PostError)

    	PostAster = import_module(SimMaster.PostAsterFile)
    	if hasattr(PostAster, 'Combined'):
    		VL.Logger('Combined function started', Print=True)

    		if VL.mode in ('Interactive','Terminal'):
    			err = PostAster.Combined(VL)
    		else :
    			with open(VL.LogFile, 'a') as f:
    				with contextlib.redirect_stdout(f):
    					# stderr may need to be written to a seperate file and then copied over
    					with contextlib.redirect_stderr(sys.stdout):
    						err = PostAster.Combined(VL)

    		if err == None:
    			VL.Logger('Combined function completed successfully', Print=True)
    		else :
    			VL.Exit("Combined function returned error '{}'".format(err))

    	VL.Logger('>>> PostAster Stage Complete\n', Print=True)

    VL.Logger('### Simulations Completed ###',Print=True)

    # Opens up all results in ParaVis
    if ShowRes:
    	print("### Opening .rmed files in ParaVis ###\n")
    	ResFiles = {}
    	for SimName, StudyDict in VL.SimData.items():
    		for root, dirs, files in os.walk(StudyDict['CALC_DIR']):
    			for file in files:
    				fname, ext = os.path.splitext(file)
    				if ext == '.rmed':
    					ResFiles["{}_{}".format(SimName,fname)] = "{}/{}".format(root, file)
    	Script = "{}/VLPackages/Salome/ShowRes.py".format(VL.COM_SCRIPTS)
    	VL.Salome.Run(Script, GUI=True, ArgDict=ResFiles)
