import os
import sys
sys.dont_write_bytecode=True
from types import SimpleNamespace as Namespace
from importlib import import_module
import copy
import pickle

import Scripts.Common.VLFunctions as VLF
from Scripts.Common.VLParallel import VLPool

'''
DA - Data Analysis
'''

def Setup(VL, RunDA=True):
    VL.SIM_DA = "{}/DA".format(VL.SIM_SCRIPTS)
    VL.DAData = {}
    DADicts = VL.CreateParameters(VL.Parameters_Master, VL.Parameters_Var,'DA')

    # if either DADicts is empty or RunDA is False we will return
    if not (RunDA and DADicts): return

    VL.tmpDA_DIR = "{}/DA".format(VL.TEMP_DIR)
    os.makedirs(VL.tmpDA_DIR, exist_ok=True)

    for DAName, ParaDict in DADicts.items():
        if not os.path.isfile('{}/{}.py'.format(VL.SIM_DA,ParaDict['File'])):
            VL.Exit(VLF.ErrorMessage("The file {}/{}.py does not "\
                    "exist".format(VL.SIM_DA,ParaDict['File'])))

        CALC_DIR = "{}/{}".format(VL.PROJECT_DIR, DAName)
        DADict = {'Name':DAName,
                 'CALC_DIR':CALC_DIR,
                 'TMP_CALC_DIR':"{}/{}".format(VL.tmpDA_DIR, DAName),
                 'Parameters':Namespace(**ParaDict),
                 'Data':{}}

        # Important information can be added to Data during any stage of the
        # data analysis, and this will be saved to the location specified by the
        # value for the __file__ key
        DADict['Data'] = {'__file__':"{}/Data.pkl".format(DADict['CALC_DIR'])}

        if VL.mode in ('Headless','Continuous'):
            DADict['LogFile'] = "{}/Output.log".format(DADict['CALC_DIR'])
        else : DADict['LogFile'] = None

        os.makedirs(DADict["TMP_CALC_DIR"],exist_ok=True)

        VL.DAData[DAName] = DADict

def PoolRun(VL, DADict):

    Parameters = DADict["Parameters"]

    os.makedirs(DADict['CALC_DIR'], exist_ok=True)
    VLF.WriteData("{}/Parameters.py".format(DADict['CALC_DIR']), Parameters)

    DAmod = import_module(Parameters.File)
    DASgl = getattr(DAmod, 'Single', None)
    err = DASgl(VL,DADict)
    return err

def Run(VL):
    if not VL.DAData: return
    sys.path.insert(0,VL.SIM_DA)

    VL.Logger('\n### Starting Data Analysis ###\n', Print=True)

    NbDA = len(VL.DAData)
    DADicts = list(VL.DAData.values())

    Errorfnc = VLPool(VL,PoolRun,DADicts)
    if Errorfnc:
        VL.Exit(VLF.ErrorMessage("The following DA routine(s) finished with errors:\n{}".format(Errorfnc)))

    VL.Logger('\n### Data Analysis Complete ###',Print=True)
