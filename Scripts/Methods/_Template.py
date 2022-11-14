import os
import sys
sys.dont_write_bytecode=True
from types import SimpleNamespace as Namespace

from Scripts.Common.VLParallel import VLPool
from Scripts.Common.utils import Method_base
'''
Template file for creating a new method. Create a copy of this file as #MethodName.py,
which is the name of the new method, and edit as desired. Any file starting with
_ are ignored.
'''

class Method(Method_base):
    def __init__(self,VL):
        '''
        Otional class initiation function. If this is used you will also need
        initiate Method_base using 'super().__init__(VL)' (see Mesh method for
        more details).
        '''

    def Setup(self, VL, MethodDicts, Import=False):
        '''
        Functions used for setting things up. Parameters associated with the method
        name for the Parameters_Master and Var files are passed as the
        'MethodDicts' argument.

        Information is assigned to the dictionary self.Data for use in the self.Run
        and self.PoolRun functions. See the other available methods for examples.

        If the flag for running a method, set using Run#MethodName in VirtualLab.Parameters,
        or #MethodName is not included in the parameters file(s) you will likely
        want to skip this function. Add

        'if not (self.RunFlag and MethodDicts): return'

        at the top of this file to return immediately.


        '''

        if not (self.RunFlag and MethodDicts): return

        for MethodName,MethodParameters in MethodDicts.items():
            # Perform some checks on the info in MethodParams

            '''
            Create a dictionary containing the parameters and other useful
            information for use by the PoolRun function. This information is
            assigned to self.Data.
            '''
            AnalysisDict = { 'Parameters' : Namespace(**MethodParameters),
                           # Add other useful info here also
                         }
            self.Data[MethodName] = AnalysisDict

    @staticmethod
    def PoolRun(VL,AnalysisDict,**kwargs):
        '''
        Functions which does something with the information from AnalysisDict.
        See Mesh, Sim, DA for more details.

        Note: This must have the decorator @staticmethod as it does not take the
        argument 'self'.
        '''


    def Run(self,VL,**kwargs):
        '''
        This is the function called when running VirtualLab.#MethodName in the
        run file. This uses the information assigned to self.Data to perform
        analysis.

        Use VLPool for high throughput parallelisation. This uses the information
        from the 'Launcher' and 'NbJobs' specified in the settings to run the analyses
        in parallel.
        Note: self.GetPoolRun() is a safer way of getting the PoolRun function.

        Check for errors and exit if there are any.
        '''
        VL.Logger('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n'\
                  '~~~ Starting #MethodName ~~~\n'\
                  '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n', Print=True)

        AnalysisDicts = list(self.Data.values()) # Data assigned during Setup

        Errorfnc = VLPool(VL,self.GetPoolRun(),AnalysisDicts)
        if Errorfnc:
            VL.Exit(VLF.ErrorMessage("The following #MethodName routine(s) finished with errors:\n{}".format(Errorfnc)),
                    Cleanup=False)


        VL.Logger('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n'\
                  '~~~ #MethodName Complete ~~~\n'\
                  '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n',Print=True)
