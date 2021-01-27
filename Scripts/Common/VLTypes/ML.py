
import os
import sys
sys.dont_write_bytecode=True
from types import SimpleNamespace as Namespace
from importlib import import_module

def Setup(VL, **kwargs):
    VL.MLData = {}
    MLDicts = VL.CreateParameters(VL.Parameters_Master, VL.Parameters_Var,'ML')

    # if either MLDicts is empty or RunML is False we will return
    if not (kwargs.get('RunML', True) and MLDicts): return

    VL.tmpML_DIR = "{}/ML".format(VL.TEMP_DIR)
    os.makedirs(VL.tmpML_DIR, exist_ok=True)

    VL.ML_DIR = "{}/ML".format(VL.PROJECT_DIR)
    os.makedirs(VL.ML_DIR, exist_ok=True)

    for MLName, ParaDict in MLDicts.items():

        Mdict = {'CALC_DIR':"{}/{}".format(VL.ML_DIR, MLName),
                 'TMP_CALC_DIR':"{}/{}".format(VL.tmpML_DIR, MLName),
                 'Parameters':Namespace(**ParaDict)}

        os.makedirs(Mdict["CALC_DIR"], exist_ok=True)
        os.makedirs(Mdict["TMP_CALC_DIR"])

        VL.MLData[MLName] = Mdict

def devRun(VL,**kwargs):
    if not VL.MLData: return
    sys.path.insert(0,VL.SIM_ML)
    for Name, MLdict in VL.MLData.items():
    	MLmod = import_module(MLdict["Parameters"].File)
    	MLmod.main(VL, MLdict)
