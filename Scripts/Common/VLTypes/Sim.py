
import os
import sys
sys.dont_write_bytecode=True
from types import SimpleNamespace as Namespace
from importlib import import_module
from pathos.multiprocessing import ProcessPool
from ..VLPackages import CodeAster

class Sim():
    def __init__(self,VL):
        self.VL = VL
        self.Data = {}
        self.CodeAster = CodeAster.CodeAster(VL)

    def Setup(self,**kwargs):
        os.makedirs(self.VL.STUDY_DIR, exist_ok=True)
        MetaInfo = {key:val for key,val in self.VL.__dict__.items() if type(val)==str}
        SimDicts = self.VL.CreateParameters(self.VL.Parameters_Master,self.VL.Parameters_Var,'Sim')
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
            StudyDict['TMP_CALC_DIR'] = TMP_CALC_DIR = "{}/{}".format(self.VL.TMP_DIR, SimName)
            StudyDict['CALC_DIR'] = CALC_DIR = "{}/{}".format(self.VL.STUDY_DIR, SimName)
            StudyDict['PREASTER'] = "{}/PreAster".format(CALC_DIR)
            StudyDict['ASTER'] = "{}/Aster".format(CALC_DIR)
            StudyDict['POSTASTER'] = "{}/PostAster".format(CALC_DIR)
            StudyDict['MeshFile'] = "{}/{}.med".format(self.VL.MESH_DIR, ParaDict['Mesh'])

            # Create tmp directory & add __init__ file so that it can be treated as a package
            if not os.path.isdir(TMP_CALC_DIR): os.makedirs(TMP_CALC_DIR)
            with open("{}/__init__.py".format(TMP_CALC_DIR),'w') as f: pass
            # Combine Meta information with that from Study dict and write to file for salome/CodeAster to import
            self.VL.WriteModule("{}/PathVL.py".format(TMP_CALC_DIR), {**MetaInfo, **StudyDict})
            # Write Sim Parameters to file for Salome/CodeAster to import
            self.VL.WriteModule("{}/Parameters.py".format(TMP_CALC_DIR), ParaDict)

            # Attach Parameters to StudyDict for ease of access
            StudyDict['Parameters'] = Namespace(**ParaDict)
            # Add StudyDict to SimData dictionary
            self.Data[SimName] = StudyDict.copy()

            self.MeshNames.append(ParaDict['Mesh'])

    def PoolRun(self, StudyDict,**kwargs):
        os.makedirs(StudyDict['ASTER'],exist_ok=True)
        # Create export file for CodeAster
        ExportFile = "{}/Export".format(StudyDict['ASTER'])
        CommFile = '{}/{}.comm'.format(self.VL.SIM_ASTER,StudyDict['Parameters'].AsterFile)
        MessFile = '{}/AsterLog'.format(StudyDict['ASTER'])
        self.CodeAster.ExportWriter(ExportFile, CommFile,
        							StudyDict["MeshFile"],
        							StudyDict['ASTER'], MessFile)


        # if self.mode == 'Headless': Outfile='/dev/null'
        # elif self.mode == 'Continuous': Outfile=SimLogFile.format(StudyDict['ASTER'])
        # else : Outfile=''
        Outfile=''

        SubProc = self.CodeAster.Run(ExportFile, OutFile=Outfile, AddPath=[self.VL.TMP_DIR,StudyDict['TMP_CALC_DIR']])
        SubProc.wait()


    def Run(self,**kwargs):
        RunPreAster = kwargs.get('RunPreAster',True)
        RunAster = kwargs.get('RunAster', True)
        RunPostAster = kwargs.get('RunPostAster', True)
        ShowRes = kwargs.get('ShowRes', False)
        mpi_nbcpu = kwargs.get('mpi_nbcpu',1)
        mpi_nbnoeud = kwargs.get('mpi_nbnoeud',1)
        ncpus = kwargs.get('ncpus',1)
        memory = kwargs.get('memory',2)
        NumThreads = kwargs.get('NumThreads',1)

        if RunAster and hasattr(self.VL.Parameters_Master.Sim,'AsterFile'):

            Arg0 = [self]*len(self.Data)
            Arg1 = list(self.Data.values())

            pool = ProcessPool(nodes=NumThreads)
            pool.map(self.PoolRun, Arg1)
