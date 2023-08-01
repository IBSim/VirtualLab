import os
import sys
import shutil

import numpy as np
import matplotlib.pyplot as plt

from Scripts.Common.ML import ML, GPR, NN
from Scripts.Common.tools import MEDtools
from Scripts.Common.tools.MED4Py import WriteMED
from Scripts.VLPackages.ParaViS import API as ParaViS
from Scripts.VLPackages.ERMES.ERMESFunc import Variation

dirname = os.path.dirname(os.path.abspath(__file__))

def _MakeMEDResult(MeshFile,ResFile,FieldResult={}):

    mesh = MEDtools.MeshInfo(MeshFile)
    CoilFace = mesh.GroupInfo('CoilFace')
    NodeIDs = CoilFace.Nodes
    mesh.Close()

    shutil.copy(MeshFile,ResFile)
    res_obj = WriteMED(ResFile,append=True)
    for ResName,values in FieldResult.items():
        values_full = np.zeros(mesh.NbNodes)
        values_full[NodeIDs-1] = values
        res_obj.add_nodal_result(values_full,ResName)

def CreateImage_GPR(VL,DataDict):

    Parameters = DataDict['Parameters']
    MLModel = Parameters.MLModel
    model_path = "{}/{}".format(VL.ML.output_dir,MLModel)
    model = GPR.GetModelPCA(model_path) # load in model

    _CreateImage(VL,DataDict,model)
    _Variation(VL,DataDict,model)

def CreateImage_MLP(VL,DataDict):

    Parameters = DataDict['Parameters']
    MLModel = Parameters.MLModel
    model_path = "{}/{}".format(VL.ML.output_dir,MLModel)
    model = NN.GetModelPCA(model_path) # load in model

    _CreateImage(VL,DataDict,model)
    _Variation(VL,DataDict,model)

def _Variation(VL,DataDict,model):

    Parameters = DataDict['Parameters']
    TestIn, TestOut = ML.VLGetDataML(VL,Parameters.TestData)
    meshfile = "{}/{}.med".format(VL.MESH_DIR,Parameters.MeshName)

    mesh = MEDtools.MeshInfo(meshfile)
    CoilFace = mesh.GroupInfo('CoilFace')
    CoilFaceCoords = mesh.GetNodeXYZ(CoilFace.Nodes)

    prediction = model.PredictFull(TestIn)
    pred_variation = Variation(prediction,CoilFace.Connect,CoilFaceCoords,CoilFace.Nodes)
    true_variation = Variation(TestOut,CoilFace.Connect,CoilFaceCoords,CoilFace.Nodes)

    metrics = ML.GetMetrics(pred_variation,true_variation)

    print()
    print('Variation prediction using Joule heating prediction on surface\n')
    print(metrics.sum())

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
    # add results to dictionary to create results MED file
    FieldResults = {}
    for ix,simulation,mlmodel in zip(Index,TestOut[Index],prediction):
        FieldResults['Simulation_{}'.format(ix)] = simulation
        FieldResults['ML_{}'.format(ix)] = mlmodel

    _MakeMEDResult(meshfile,resfile_tmp,FieldResult=FieldResults)

    PVFile = '{}/ParaViS.py'.format(dirname)
    func_evals = []
    for ix in Index:
        funcname = 'HeatingProfileCompare' # func from PVFile which is called
        arg1 = resfile_tmp # path to the med file
        arg2 = ['ML_{}'.format(ix),'Simulation_{}'.format(ix)] # name of the results to compare
        arg3 = ["{}/Ex{}_ML.png".format(DataDict['CALC_DIR'],ix),"{}/Ex{}_Simulation.png".format(DataDict['CALC_DIR'],ix)]
        arg4 = "{}/Ex{}_Error.png".format(DataDict['CALC_DIR'],ix)
        func_evals.append([funcname,(arg1,arg2,arg3,arg4)])

    ParaViS.RunEval(PVFile,func_evals,GUI=True)

