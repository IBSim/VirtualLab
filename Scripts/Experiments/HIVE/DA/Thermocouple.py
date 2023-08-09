import os
import shutil
from importlib import import_module

import numpy as np
import matplotlib.pyplot as plt

from Scripts.Common.ML import ML, GPR, NN
from Scripts.Common.Optimisation import optimisation
from Scripts.Common.tools.MED4Py import WriteMED
from Scripts.VLPackages.ParaViS import API as ParaViS
from Scripts.Common.tools import MEDtools
from Scripts.Common import VLFunctions as VLF

dirname = os.path.dirname(os.path.abspath(__file__))
PVFile = '{}/ParaViS.py'.format(dirname)

def FullFieldEstimate_GPR(VL,DataDict):
    '''
    Routine which uses GPR model to estimate full field. See _FullFieldEstimate for more details
    '''
    Parameters = DataDict['Parameters']
    MLModel = Parameters.MLModel
    model = GPR.GetModelPCA("{}/{}".format(VL.ML.output_dir,MLModel)) # load temperature model

    _FullFieldEstimate(VL,DataDict,model)

def FullFieldEstimate_MLP(VL,DataDict):
    '''
    Routine which uses MLP model to estimate full field. See _FullFieldEstimate for more details
    '''
    Parameters = DataDict['Parameters']
    MLModel = Parameters.MLModel
    model = NN.GetModelPCA("{}/{}".format(VL.ML.output_dir,MLModel)) # load temperature model

    _FullFieldEstimate(VL,DataDict,model)

def _FullFieldEstimate(VL,DataDict,model):
    '''
    Routine to predict the temperature field using the thermocouples outlined 
    in ThemocoupleConfig. The estimated temperature field is then compared with 
    the simulation to assess its accuracy. 
    '''
    Parameters = DataDict['Parameters']
    # load in simulation data and extract results specified by Index
    TestData = Parameters.TestData
    Index = Parameters.Index
    TestIn, TestOut = ML.VLGetDataML(VL,TestData)
    simulated_temp = TestOut[Index]
    # get full path to meshfile
    MeshName = Parameters.MeshName
    meshfile = "{}/{}.med".format(VL.MESH_DIR,MeshName)
    # calculate the temperatures at the thermocouple locations from the full field simulation data
    TC_config = Parameters.ThermocoupleConfig
    interpolation = _InterpolateTC(TC_config,meshfile) # identify the element which each thermocouple belongs to
    temp_at_TC =_TCValues(interpolation,simulated_temp)
    # calculate the experimental parameters which deliver the thermocouple temperatures
    exp_parameters,val = _InverseTC_multi(model,temp_at_TC,interpolation)
    # pass the experimental parameters back to the model to predict the full temperature field
    # and plot images comparing it to the simulated temperature field
    estimated_field = model.PredictFull(exp_parameters)
    resfile_tmp = "{}/compare.med".format(DataDict['TMP_CALC_DIR'])
    shutil.copy(meshfile,resfile_tmp)
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

def Sensitivity_GPR(VL,DataDict):
    '''
    Routine which uses GPR model to estimate full field. See _Sensitivity for more details
    '''
    Parameters = DataDict['Parameters']
    MLModel = Parameters.MLModel
    model = GPR.GetModelPCA("{}/{}".format(VL.ML.output_dir,MLModel)) # load temperature model
    _Sensitivity(VL,DataDict,model)

def Sensitivity_MLP(VL,DataDict):
    '''
    Routine which uses MLP model to estimate full field. See _Sensitivity for more details
    '''
    Parameters = DataDict['Parameters']
    MLModel = Parameters.MLModel
    model = NN.GetModelPCA("{}/{}".format(VL.ML.output_dir,MLModel)) # load temperature model
    _Sensitivity(VL,DataDict,model)

def _Sensitivity(VL,DataDict,model):
    '''
    Routine to highlight the effect that the thermocouple placement has on the ability
    to accurately predict a unique temperature field. 
    '''
    Parameters = DataDict['Parameters']
    # load the simulation data and keep only a few to work out the average number of 
    # temperature fields a configuration of thermocouples will result in 
    TestData = Parameters.TestData
    nbexample = getattr(Parameters,'NbExample',5)
    TestIn, TestOut = ML.VLGetDataML(VL,TestData)
    TestOut = TestOut[:nbexample]
    # get full path of mesh file
    MeshName = Parameters.MeshName
    meshfile = "{}/{}.med".format(VL.MESH_DIR,MeshName)
    # set a default seed for reproducibility
    seed = getattr(Parameters,'Seed',100)
    np.random.seed(seed)

    # ===============================================================================
    # create NbConfig number of random configurations of thermocouples on the CandidateSurfaces
    # and calculate the average number of admissible temperature fields (the lower, the better)
    CandidateSurfaces = Parameters.CandidateSurfaces
    NbTC = Parameters.NbThermocouples
    NbConfig = Parameters.NbConfig
    NbInit = 50
    config_score,tc_configs = [],[]
    x = list(range(NbConfig))
    for _ in x:
        TC_config = _random_tc_config(CandidateSurfaces,NbTC)
        interpolation = _InterpolateTC(TC_config,meshfile)
        nb_field = []
        for temp_field in TestOut:
            temp_at_TC =_TCValues(interpolation,temp_field)
            cd,val = _InverseTC(model,temp_at_TC,interpolation,NbInit=NbInit)
            unique_parameters,unique_field = _UniqueSol(model,cd,meshfile)
            _nb_field = len(unique_parameters)
            if _nb_field!=0: 
                nb_field.append(_nb_field)
        config_score.append(np.mean(nb_field))
        tc_configs.append(TC_config)

    # ===============================================================================
    # create a plot showing the average number of admissible fields for each configuration
    plt.figure()
    plt.scatter(x,config_score,c='k',marker='x')
    x_ticks = ["Config_{}".format(i+1) for i in x]
    plt.xticks(x,x_ticks)
    plt.ylabel("No. admissible fields (avg)")
    plt.title("Variation in number of admissible fields\nfor {} randomly placed thermocouples".format(NbTC))
    plt.savefig("{}/PlacementSensitivity.png".format(DataDict['CALC_DIR']))
    plt.close()

    # ===============================================================================
    # plot the locations of the thermocouples
    # define directory to put images in and delete it if it already exists
    config_dir = "{}/TC_configs".format(DataDict['CALC_DIR'])
    if os.path.isdir(config_dir): shutil.rmtree(config_dir)
    # convert the thermocouple locations to actual coordinates and pass to PlotTC paraview function
    paravis_evals = []
    for tc_config,name in zip(tc_configs,x_ticks):
        tc_coords = _LocationTC(tc_config,meshfile)
        image_dir = "{}/{}".format(config_dir,name)
        os.makedirs(image_dir)
        args = [meshfile,tc_coords,image_dir]
        paravis_evals.append(['PlotTC',args])

    ParaViS.RunEval(PVFile,paravis_evals,GUI=True)


    # resfile_tmp = "{}/compare.med".format(DataDict['TMP_CALC_DIR'])
    # shutil.copy(meshfile,resfile_tmp)
    # dict1 = {'sol_{}'.format(i):val for i,val in enumerate(unique_field)}
    # _AddResult(resfile_tmp,**dict1)
    # ParaViS.ShowMED(resfile_tmp,GUI=True)

def _LocationTC(TC_config,meshfile):
    # returns the coordinates of the thermocouples
    interpolation = _InterpolateTC(TC_config,meshfile)
    mesh = MEDtools.MeshInfo(meshfile)
    points = []
    for nodes,weights in interpolation:
        coords = mesh.GetNodeXYZ(nodes)
        cd = (coords*weights[:,None]).sum(axis=0)
        points.append(cd)
    mesh.Close()

    return points

def _UniqueSol(mod,inverse_sol,meshfile,diff_frac=0.025):
    if len(inverse_sol)==0: return [],[]

    # get the indexes associated with the mesh excluding the coil
    mesh = MEDtools.MeshInfo(meshfile)
    Pipe = mesh.GroupInfo('Pipe')
    mesh_nodes = list(range(1,mesh.NbNodes+1))
    use_ix = np.array(list(set(mesh_nodes).difference(Pipe.Nodes))) -1 
    mesh.Close()

    preds = mod.PredictFull(inverse_sol)
    preds_nopipe = preds[:,use_ix]

    UniquePred,keep = preds_nopipe[:1],[0]
    for i,pred in enumerate(preds_nopipe[1:]):
        diff_sc = np.abs(UniquePred - pred)/pred
        diff_mean = diff_sc.mean(axis=1)# mean absolute percentage difference

        if (diff_mean > diff_frac).all():
            #only keep ones which are different to the others
            UniquePred = np.vstack((UniquePred,pred))
            keep.append(i+1)

    return inverse_sol[keep], preds[keep]


def _random_tc_config(Surfaces,NbTC):
    nb_surface = len(Surfaces)
    config = []
    for _ in range(NbTC):
        surf_name = Surfaces[np.random.randint(nb_surface)]
        position = np.random.uniform(0,1,size=2)
        config.append([surf_name,*position])
    return config

def _AddResult(ResFile,**kwargs):
    res_obj = WriteMED(ResFile,append=True)
    for ResName,values in kwargs.items():
        res_obj.add_nodal_result(values,ResName)
    res_obj.close()

def _InverseTC_multi(model,target_tc,interpolation):
    cd,val = [],[]
    for _target_tc in target_tc:
        _cd,_val = _InverseTC(model,_target_tc,interpolation)
        cd.append(_cd[0]);val.append(_val[0])
    return np.array(cd),np.array(val)

def _InverseTC(model,target_tc,interpolation,NbInit=20,seed=100):
    bounds = [[0,1]]*model.Dataspace.NbInput
    cd_scale, val, val_lse = optimisation.GetOptimaLSE(_field_TC,target_tc,NbInit,bounds,seed=seed,fnc_args = [model,interpolation])
    cd = model.RescaleInput(cd_scale) # rescale back from [0,1] range
    return cd, val

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

def _GetNorm(MeshFile,SurfaceName):
    ''' Get norm to surface from mesh creation file'''
    MeshParameters = VLF.ReadParameters("{}.py".format(os.path.splitext(MeshFile)[0]))
    Mesh_File = import_module("Mesh.{}".format(MeshParameters.File))
    SurfaceNormals = Mesh_File.SurfaceNormals
    norm = SurfaceNormals[SurfaceNormals[:,0]==SurfaceName,1]
    return norm

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

