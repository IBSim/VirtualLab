import os
import sys
sys.dont_write_bytecode=True
from types import SimpleNamespace as Namespace

import Scripts.Common.VLFunctions as VLF
from Scripts.Common.VLParallel import VLPool
from Scripts.Common.VLTypes import Method_base
'''
DA - Data Analysis
'''
class Method(Method_base):
    def Setup(self, VL, RunDA=True, Import=False):
        DADicts = VL.CreateParameters(VL.Parameters_Master, VL.Parameters_Var,'DA')

        # if either DADicts is empty or RunDA is False we will return
        if not (RunDA and DADicts): return

        VL.tmpDA_DIR = "{}/DA".format(VL.TEMP_DIR)
        os.makedirs(VL.tmpDA_DIR, exist_ok=True)


        FileDict = {}
        for DAName, ParaDict in DADicts.items():
            CALC_DIR = "{}/{}".format(VL.PROJECT_DIR, DAName)
            if Import:
                ParaDict = VLF.ImportUpdate("{}/Parameters.py".format(CALC_DIR), ParaDict)

            # ======================================================================
            # get file path & perform checks
            # default name is Single
            file_name,func_name = VLF.FileFuncSplit(ParaDict['File'],'Single')

            if (file_name,func_name) not in FileDict:
                # Check file in directory & get path
                FilePath = VL.GetFilePath([VL.SIM_DA,VL.VLRoutine_SCRIPTS], file_name,
                                          file_ext='py', exit_on_error=True)
                # Check function func_name is in the file
                VL.GetFunction(FilePath,func_name,exit_on_error=True)
                File_func = [FilePath,func_name]
                FileDict[(file_name,func_name)] = File_func
            else:
                File_func = FileDict[(file_name,func_name)]

            # ==========================================================================
            # Create dictionary for each analysis
            DADict = {'Name':DAName,
                     'CALC_DIR':CALC_DIR,
                     'TMP_CALC_DIR':"{}/{}".format(VL.tmpDA_DIR, DAName),
                     'Parameters':Namespace(**ParaDict),
                     'FileInfo':File_func,
                     'Data':{}}

            # Important information can be added to Data during any stage of the
            # data analysis, and this will be saved to the location specified by the
            # value for the __file__ key
            DADict['Data'] = {'__file__':"{}/Data.pkl".format(DADict['CALC_DIR'])}

            if VL.mode in ('Headless','Continuous'):
                DADict['LogFile'] = "{}/Output.log".format(DADict['CALC_DIR'])
            else : DADict['LogFile'] = None

            os.makedirs(DADict["TMP_CALC_DIR"],exist_ok=True)

            self.Data[DAName] = DADict

    @staticmethod
    def PoolRun(VL, DADict):

        Parameters = DADict["Parameters"]

        os.makedirs(DADict['CALC_DIR'], exist_ok=True)
        VLF.WriteData("{}/Parameters.py".format(DADict['CALC_DIR']), Parameters)

        DASgl = VLF.GetFunc(*DADict['FileInfo'])

        err = DASgl(VL,DADict)

        return err

    def Run(self,VL):
        if not self.Data: return
        sys.path.insert(0,VL.SIM_DA)

        VL.Logger('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n'\
                  '~~~ Starting Data Analysis ~~~\n'\
                  '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n', Print=True)

        NbDA = len(self.Data)
        DADicts = list(self.Data.values())

        Errorfnc = VLPool(VL,self.PoolRun,DADicts)
        if Errorfnc:
            VL.Exit(VLF.ErrorMessage("The following DA routine(s) finished with errors:\n{}".format(Errorfnc)),
                    Cleanup=False)

        VL.Logger('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n'\
                  '~~~ Data Analysis Complete ~~~\n'\
                  '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n',Print=True)
