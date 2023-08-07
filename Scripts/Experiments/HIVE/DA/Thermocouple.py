import os
import shutil
from importlib import import_module

import numpy as np

from Scripts.Common.ML import ML, GPR, NN
from Scripts.Common.Optimisation import optimisation
from Scripts.Common.tools.MED4Py import WriteMED
from Scripts.VLPackages.ParaViS import API as ParaViS
from Scripts.Common.tools import MEDtools
from Scripts.Common import VLFunctions as VLF

dirname = os.path.dirname(os.path.abspath(__file__))
PVFile = '{}/ParaViS.py'.format(dirname)

def FullFieldEstimate(VL,DataDict):

    Parameters = DataDict['Parameters']
    TestData = Parameters.TestData
    Index = Parameters.Index
    TC_config = Parameters.ThermocoupleConfig
    MeshName = Parameters.MeshName

    TestIn, TestOut = ML.VLGetDataML(VL,TestData)

    meshfile = "{}/{}.med".format(VL.MESH_DIR,MeshName)
    
    MLModel = Parameters.MLModel
    model = GPR.GetModelPCA("{}/{}".format(VL.ML.output_dir,MLModel)) # load temperature model

    _Compare(model,meshfile,TC_config,TestOut,Index,DataDict)


def _AddResult(ResFile,**kwargs):
    res_obj = WriteMED(ResFile,append=True)
    for ResName,values in kwargs.items():
        res_obj.add_nodal_result(values,ResName)

def _Compare(model,meshfile,TC_config,simulated_temp,Index,DataDict):

    resfile_tmp = "{}/compare.med".format(DataDict['TMP_CALC_DIR'])
    shutil.copy(meshfile,resfile_tmp)

    interpolation = _InterpolateTC(TC_config,meshfile)
    simulated_temp = simulated_temp[Index]
    temp_at_TC =_TCValues(interpolation,simulated_temp)

    cd,val = _InverseTC_multi(model,temp_at_TC,interpolation)

    estimated_field = model.PredictFull(cd)

    paravis_evals = []
    for ml,sim,ix in zip(estimated_field,simulated_temp,Index):
        ml_name,sim_name = "ML_{}".format(ix),"Simulation_{}".format(ix)
        _AddResult(resfile_tmp,**{ml_name:ml,sim_name:sim})

        arg1 = resfile_tmp # path to the med file
        arg2 = [ml_name,sim_name] # name of the results to compare
        arg3 = ["{}/Ex{}_ML.png".format(DataDict['CALC_DIR'],ix),"{}/Ex{}_Simulation.png".format(DataDict['CALC_DIR'],ix)]
        arg4 = "{}/Ex{}_Error.png".format(DataDict['CALC_DIR'],ix)
        paravis_evals.append(['TemperatureCompare',
                              (arg1,arg2,arg3,arg4)])

    ParaViS.RunEval(PVFile,paravis_evals,GUI=True)

def _InverseTC_multi(model,target_tc,interpolation):
    cd,val = [],[]
    for _target_tc in target_tc:
        _cd,_val = _InverseTC(model,_target_tc,interpolation)
        cd.append(_cd[0]);val.append(_val[0])
    return np.array(cd),np.array(val)

def _InverseTC(model,target_tc,interpolation):
    bounds = [[0,1]]*model.Dataspace.NbInput
    cd_scale, val, val_lse = optimisation.GetOptimaLSE(_field_TC,target_tc,10,bounds,seed=100,fnc_args = [model,interpolation])
    cd = model.RescaleInput(cd_scale) # rescale back from [0,1] range
    return cd, val




def _GetNorm(MeshFile,SurfaceName):
    ''' Get norm to surface from mesh creation file'''
    MeshParameters = VLF.ReadParameters("{}.py".format(os.path.splitext(MeshFile)[0]))
    Mesh_File = import_module("Mesh.{}".format(MeshParameters.File))
    SurfaceNormals = Mesh_File.SurfaceNormals
    norm = SurfaceNormals[SurfaceNormals[:,0]==SurfaceName,1]
    return norm

def _GetInterp(MeshFile,SurfaceName,x1,x2):
    ''' Get the node index & weights to inteprolate value at a point on the
    surface of the sample for TC measurements.'''

    # Get coordinates of the group
    meshdata = MEDtools.MeshInfo(MeshFile)
    group = meshdata.GroupInfo(SurfaceName)
    Coords = meshdata.GetNodeXYZ(group.Nodes)

    # Know which coordinates to keep based on the surface normal
    norm = _GetNorm(MeshFile,SurfaceName)
    if norm == 'NX': Coords = Coords[:,[1,2]] 
    elif norm == 'NY': Coords = Coords[:,[0,2]]
    elif norm == 'NZ': Coords = Coords[:,[0,1]]

    # scale coordinates to [0,1] range
    cd_min, cd_max = Coords.min(axis=0),Coords.max(axis=0)
    Coords = (Coords - cd_min)/(cd_max - cd_min)

    # Find nodes & weights to interpolate value at x1,x2
    nodes,weights = VLF.Interp_2D(Coords,group.Connect,(x1,x2))

    meshdata.Close()

    return nodes, weights

def _TCValues(interpolation,nodal_data):
    tc_target = []
    for ixs,weights in interpolation:

        if nodal_data.ndim==2:
            tc_T = (nodal_data[:,ixs]*weights).sum(axis=1)
        else:
            tc_T = (nodal_data[ixs]*weights).sum()
        tc_target.append(tc_T)
    return np.array(tc_target).T

def _InterpolateTC(TCData,meshfile):
    ''' Get nodes indexes & weights for all thermocouples provided'''
    Interp = [_GetInterp(meshfile,SurfName,x1,x2) for SurfName,x1,x2 in TCData]
    return Interp

def _field_TC(X,mod,interpolation):
    pred,grad = mod.Gradient(X,scale_inputs=False)

    TC_pred,TC_grad = [],[]
    for ixs,weights in interpolation:
        # get prediction and gradient on the nodes which make up the element the thermocouple is within
        pred_ixs = mod.Reconstruct(pred,index=ixs)
        grad_ixs = mod.ReconstructGradient(grad,index=ixs)
        # interpolate the value to the exact point
        pred_interp = (pred_ixs*weights).sum(axis=1)
        grad_interp = np.einsum('ijk,j->ik',grad_ixs,weights)
        TC_pred.append(pred_interp); TC_grad.append(grad_interp)
    # ensure pred and grad are in the corretc shape before returning
    TC_pred = np.transpose(TC_pred)
    TC_grad = np.moveaxis(TC_grad,0,1)
    return TC_pred,TC_grad

# def _field_TC2(X,mod,interpolation):
#     pred,grad = mod.Gradient(X,scale_inputs=False)

#     TC_pred,TC_grad = [],[]
#     for ixs,weights in interpolation:

#         _pred_full = pred.dot(mod.VT[:,ixs])
#         _pred_full = ML.DataRescale(_pred_full,*mod.ScalePCA[:,ixs])
#         a = (_pred_full*weights).sum(axis=1)

#         FullGrad = []
#         for i in range(mod.Dataspace.NbInput):
#             _grad = grad[:,:,i].dot(mod.VT[:,ixs])
#             _grad = ML.DataRescale(_grad,0,mod.ScalePCA[1,ixs]) # as its gradient we set the bias term to zero

#             _grad = (_grad*weights).sum(axis=1)
#             FullGrad.append(_grad)
#         FullGrad = np.moveaxis(FullGrad, 0, -1)

#         TC_pred.append(a); TC_grad.append(FullGrad)
#     TC_pred = np.transpose(TC_pred)
#     TC_grad = np.moveaxis(TC_grad,0,1)
#     print(TC_grad[1,0])
#     print()
#     return TC_pred,TC_grad