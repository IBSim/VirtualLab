import os
import sys
import shutil

import numpy as np

from Scripts.Common.ML import ML, GPR, NN
from Scripts.Common.tools.MED4Py import WriteMED
from Scripts.Common.tools import MEDtools
from Scripts.VLPackages.ParaViS import API as ParaViS

dirname = os.path.dirname(os.path.abspath(__file__))

def _MakeMEDResult(MeshFile,ResFile,FieldResults={}):
    shutil.copy(MeshFile,ResFile)
    res_obj = WriteMED(ResFile,append=True)
    for ResName,values in FieldResults.items():
        res_obj.add_nodal_result(values,ResName)


def CreateImage_GPR(VL,DataDict):

    Parameters = DataDict['Parameters']
    MLModel = Parameters.MLModel
    model_path = "{}/{}".format(VL.ML.output_dir,MLModel)
    model = GPR.GetModelPCA(model_path) # load in model

    # _CreateImage(VL,DataDict,model)
    _InverseSolution(VL,DataDict,model)

def CreateImage_MLP(VL,DataDict):

    Parameters = DataDict['Parameters']
    MLModel = Parameters.MLModel
    model_path = "{}/{}".format(VL.ML.output_dir,MLModel)
    model = NN.GetModelPCA(model_path) # load in model

    # _CreateImage(VL,DataDict,model)
    _InverseSolution(VL,DataDict,model)


def _CreateImage(VL,DataDict, model):

    Parameters = DataDict['Parameters']
    MeshName = Parameters.MeshName
    TestData = Parameters.TestData
    Index = Parameters.Index

    TestIn, TestOut = ML.VLGetDataML(VL,TestData)

    resfile_tmp = "{}/temporary.med".format(DataDict['TMP_CALC_DIR'])
    meshfile = "{}/{}.med".format(VL.MESH_DIR,MeshName)

    if type(Index) != list: Index=[Index]

    # get predictions for those specified by Index
    InputIx = TestIn[Index]
    prediction = model.PredictFull(InputIx)

    FieldResults = {}
    for ix,simulation,mlmodel in zip(Index,TestOut[Index],prediction):
        FieldResults['Simulation_{}'.format(ix)] = simulation
        FieldResults['ML_{}'.format(ix)] = mlmodel

    _MakeMEDResult(meshfile,resfile_tmp,FieldResults=FieldResults)

    PVFile = '{}/ParaViS.py'.format(dirname)
    func_evals = []
    for ix in Index:
        funcname = 'TemperatureCompare' # func from PVFile which is called
        arg1 = resfile_tmp # path to the med file
        arg2 = ['ML_{}'.format(ix),'Simulation_{}'.format(ix)] # name of the results to compare
        arg3 = ["{}/Ex{}_ML.png".format(DataDict['CALC_DIR'],ix),"{}/Ex{}_Simulation.png".format(DataDict['CALC_DIR'],ix)]
        arg4 = "{}/Ex{}_Error.png".format(DataDict['CALC_DIR'],ix)
        func_evals.append([funcname,(arg1,arg2,arg3,arg4)])

    ParaViS.RunEval(PVFile,func_evals,GUI=True)

def _MaxField(X,fnc,fnc_args):
    pred,grad = fnc(X,*fnc_args)

    ix = np.argmax(pred,axis=1)
    _nb = np.arange(0,len(ix))
    pred = pred[_nb,ix]
    grad = grad[_nb,ix]
    return pred, grad

def _IS_max(model):

    bounds = [[0,1]]*model.Dataspace.NbInput
    fnc = model.GradientFull
    fnc_args = [False] # arguments for model.GradientFull (inputs are assumed to be in [0,1])
    _fnc_args = [fnc,fnc_args] # the arguments passed to _MaxField

    cd_scale,val = ML.Optimise(_MaxField,10,bounds,fnc_args=_fnc_args,seed=100)
    cd = model.RescaleInput(cd_scale)

    best_cd,best_val = cd[0], val[0]
    best_cd_str = ", ".join(["{:.2e}".format(v) for v in best_cd])
    print('###############################################\n')
    print('Parameter combination which will deliver a maximum temperature of {:.2f} C:\n'.format(best_val))
    print(best_cd_str)
    print('\n###############################################\n')

def _MaxTempInverse(model,max_temp,bounds=None,NbInit=100,seed=123):

    bounds = [[0,1]]*model.Dataspace.NbInput
    fnc = model.GradientFull
    fnc_args = [False] # arguments for model.GradientFull (inputs are assumed to be in [0,1])
    _fnc_args = [fnc,fnc_args] # the arguments passed to _MaxField

    cd_scale, val, val_lse = ML.OptimiseLSE(_MaxField, max_temp, NbInit, bounds,
                             seed=seed, fnc_args=_fnc_args)
    cd = model.RescaleInput(cd_scale)
    return cd, val, val_lse

def _MaxTemp(model):
    DesiredTemp = 600
    cd,val,val_lse = _MaxTempInverse(model,DesiredTemp)

    print('###############################################\n')
    print('Different parameter combinations which will deliver {:.2f} C:\n'.format(DesiredTemp))
    for _cd in cd[:5]:
        best_cd_str = ", ".join(["{:.2e}".format(v) for v in _cd])
        print(best_cd_str)
    print('\n###############################################\n')

def _InverseSolution(VL,DataDict, model):
    _IS_max(model)
    _MaxTemp(model)






