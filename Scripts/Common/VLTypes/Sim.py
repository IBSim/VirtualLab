
import os
import sys
sys.dont_write_bytecode=True
from types import SimpleNamespace as Namespace
from importlib import import_module
from contextlib import redirect_stderr, redirect_stdout
from pathos.multiprocessing import ProcessPool
from ..VLPackages import CodeAster

class Sim():
    def __init__(self,VL):

        self.Data = {}

        self.__dict__.update(VL.__dict__)

        # Take what's necessary from VL class
        self.Logger = VL.Logger
        self.Exit = VL.Exit
        self.WriteModule = VL.WriteModule

        self.CodeAster = CodeAster.CodeAster(VL)

    def Setup(self,SimDicts,**kwargs):
        os.makedirs(self.STUDY_DIR, exist_ok=True)
        MetaInfo = {key:val for key,val in self.__dict__.items() if type(val)==str}
        self.MeshNames = []
        for SimName, ParaDict in SimDicts.items():
            '''
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
            '''
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
            self.Data[SimName] = StudyDict.copy()

            self.MeshNames.append(ParaDict['Mesh'])

    def PoolRun(self, StudyDict, kwargs):
        RunPreAster = kwargs.get('RunPreAster',True)
        RunAster = kwargs.get('RunAster', True)
        RunPostAster = kwargs.get('RunPostAster', True)

        mpi_nbcpu = kwargs.get('mpi_nbcpu',1)
        mpi_nbnoeud = kwargs.get('mpi_nbnoeud',1)
        ncpus = kwargs.get('ncpus',1)
        memory = kwargs.get('memory',2)

        Parameters = StudyDict["Parameters"]
        SimLogFile = "{}/Output.log"

        if self.mode == 'Headless': OutFile = "{}/Output.log"
        elif self.mode == 'Continuous': OutFile = "{}/Output.log"
        else : OutFile=''

        if RunPreAster and hasattr(Parameters,'PreAsterFile'):
            sys.path.insert(0, self.SIM_PREASTER)
            PreAster = import_module(Parameters.PreAsterFile)
            PreAsterSgl = getattr(PreAster, 'Single',None)

            self.Logger("Running PreAster for '{}'\n".format(Parameters.Name),Print=True)
            os.makedirs(StudyDict['PREASTER'],exist_ok=True)

            if not OutFile:
                PreAsterSgl(self,StudyDict)
            else :
                with open(OutFile.format(StudyDict['PREASTER']),'w') as f:
                    with redirect_stdout(f), redirect_stderr(f):
                        PreAsterSgl(self,StudyDict)

        if RunAster and hasattr(Parameters,'AsterFile'):
            self.Logger("Running Aster for '{}'\n".format(Parameters.Name),Print=True)

            os.makedirs(StudyDict['ASTER'],exist_ok=True)
            # Create export file for CodeAster
            ExportFile = "{}/Export".format(StudyDict['ASTER'])
            CommFile = '{}/{}.comm'.format(self.SIM_ASTER, Parameters.AsterFile)
            MessFile = '{}/AsterLog'.format(StudyDict['ASTER'])
            self.CodeAster.ExportWriter(ExportFile, CommFile,
            							StudyDict["MeshFile"],
            							StudyDict['ASTER'], MessFile)

            if self.mode == 'Headless': AsterOut='/dev/null'
            elif self.mode == 'Continuous': AsterOut = OutFile.format(StudyDict['ASTER'])
            else : AsterOut=''

            SubProc = self.CodeAster.Run(ExportFile, OutFile=AsterOut, AddPath=[self.TMP_DIR,StudyDict['TMP_CALC_DIR']])
            SubProc.wait()
            # from subprocess import Popen
            # SubProc = Popen(['echo','Hello World'])
            # SubProc.wait()

        if RunPostAster and hasattr(Parameters,'PostAsterFile'):
            sys.path.insert(0, self.SIM_POSTASTER)
            PostAster = import_module(Parameters.PostAsterFile)
            PostAsterSgl = getattr(PostAster, 'Single', None)

            self.Logger("Running PostAster for '{}'\n".format(Parameters.Name),Print=True)
            os.makedirs(StudyDict['POSTASTER'],exist_ok=True)

            PostAsterSgl(self,StudyDict)
            if not OutFile:
                PostAsterSgl(self,StudyDict)
            else :
                with open(OutFile.format(StudyDict['POSTASTER']),'w') as f:
                    with redirect_stdout(f), redirect_stderr(f):
                        PostAsterSgl(self,StudyDict)



    def Run(self,**kwargs):
        ShowRes = kwargs.get('ShowRes', False)
        NumThreads = kwargs.get('NumThreads',1)

        # Run high throughput part in parallel
        NumSim = len(self.Data)
        NumThreads = min(NumThreads,NumSim)

        Arg1 = list(self.Data.values())
        Arg2 = [kwargs]*NumSim
        if 1:
            pool = ProcessPool(nodes=NumThreads)
            Res = pool.map(self.PoolRun, Arg1, Arg2)
        else :
            Arg0 = [self]*NumSim
            from pyina.launchers import MpiPool
            pool = MpiPool(nodes=NumThreads,source=True)
            res = pool.map(extRun, Arg0, Arg1, Arg2)
        # 
        #
        # PostAster = import_module(Parameters_Master.Sim.PostAsterFile)
        # if hasattr(PostAster, 'Combined'):
        #     self.Logger('Combined function started', Print=True)
        #     if self.mode in ('Interactive','Terminal'):
        #         err = PostAster.Combined(self)
        #     else :
        #         with open(self.LogFile, 'a') as f:
        #             with redirect_stdout(f), redirect_stderr(f):
        #                 err = PostAster.Combined(self)



# Can't use class method with pyina so create this function which calls the method
def extRun(Meta,arg1,arg2):
    Meta.PoolRun(arg1,arg2)
