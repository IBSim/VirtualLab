import os
import sys

import h5py
import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
import scipy.stats as stats
from importlib import import_module
import pathos.multiprocessing as pathosmp

import VLFunctions as VLF
from Scripts.Common.tools import MEDtools
from Scripts.Common.ML import ML
from Scripts.Common.Optimisation import slsqp_multi, GA, GA_Parallel


dtype = 'float64' # float64 is more accurate for optimisation purposes
torch_dtype = getattr(torch,dtype)
torch.set_default_dtype(torch_dtype)

# ==============================================================================
# Functions for gathering necessary data and writing to file
def CompileData(VL,DADict):
    Parameters = DADict["Parameters"]

    # ==========================================================================
    # Get list of all the results directories which will be searched through
    CmpData = Parameters.CompileData
    if type(CmpData)==str:CmpData = [CmpData]
    ResDirs = ["{}/{}".format(VL.PROJECT_DIR,resname) for resname in CmpData]

    # ==========================================================================
    # Specify the function used to gather the necessary data & any arguments required
    args= []
    if Parameters.OutputFn.lower()=="surfacetemperatures":
        OutputFn = _SurfaceTemperatures
        args = [Parameters.Surface,Parameters.InputVariables,
                Parameters.ResFileName]
    elif Parameters.OutputFn.lower()=="fieldtemperatures":
        OutputFn = _FieldTemperatures
        args = [Parameters.InputVariables, Parameters.ResFileName]

    # ==========================================================================
    # Apply OutputFn to all sub dirs in ResDirs
    InData, OutData = ML.CompileData(ResDirs,OutputFn,args=args)

    # ==========================================================================
    # Write the input and output data to DataFile_path
    DataFile_path = "{}/{}".format(VL.PROJECT_DIR,Parameters.DataFile)

    ML.WriteMLdata(DataFile_path, CmpData, Parameters.InputArray, InData,
                    attrs=getattr(Parameters,'InputAttributes',{}))
    ML.WriteMLdata(DataFile_path, CmpData, Parameters.OutputArray, OutData,
                    attrs=getattr(Parameters,'OutputAttributes',{}))

# ==============================================================================
# Create model mapping inputs to thermocouple temperatures at fixed points
def Fixed_TC(VL,DADict):
    # np.random.seed(100)
    Parameters = DADict['Parameters']

    if getattr(Parameters,'CompileData',None):
        CompileData(VL,DADict)

    # ==========================================================================
    # Get Train & test data from file DataFile_path
    DataFile_path = "{}/{}".format(VL.PROJECT_DIR, Parameters.DataFile)

    TrainIn = ML.GetMLdata2(DataFile_path, Parameters.TrainData,
                           Parameters.InputArray, getattr(Parameters,'TrainNb',-1))
    TestIn = ML.GetMLdata2(DataFile_path, Parameters.TestData,
                          Parameters.InputArray, getattr(Parameters,'TestNb',-1))

    InputAttrs = ML.GetMLattrs(DataFile_path, Parameters.TrainData,Parameters.InputArray)
    FeatureNames = InputAttrs.get('Parameters',None)

    if hasattr(Parameters,'TCLocations'):
        # Get temperature at points using surface temps
        meshfile = "{}/SampleHIVE.med".format(VL.MESH_DIR)
        NbTC = len(Parameters.TCLocations)
        NbTrain,NbTest = TrainIn.shape[0],TestIn.shape[0]
        TrainOut,TestOut = np.zeros((NbTrain,NbTC)), np.zeros((NbTest,NbTC))
        for i,(SurfaceName,x1,x2) in enumerate(Parameters.TCLocations):
            _datatrain = ML.GetMLdata2(DataFile_path, Parameters.TrainData,
                                       SurfaceName, getattr(Parameters,'TrainNb',-1))
            _datatest = ML.GetMLdata2(DataFile_path, Parameters.TestData,
                                       SurfaceName, getattr(Parameters,'TestNb',-1))
            _data = np.vstack((_datatrain,_datatest))

            interp = Get_Interp(meshfile,SurfaceName,x1,x2,_data)
            TrainOut[:,i] = interp[:NbTrain]
            TestOut[:,i] = interp[NbTrain:]

        LabelNames = "Thermocouples placed at the following locations:\n{}".format(Parameters.TCLocations)

    else:
        # This data may have been created previously
        TrainOut = ML.GetMLdata2(DataFile_path, Parameters.TrainData,
                                 Parameters.OutputArray, getattr(Parameters,'TrainNb',-1))
        TestOut = ML.GetMLdata2(DataFile_path, Parameters.TestData,
                                Parameters.OutputArray, getattr(Parameters,'TestNb',-1))

        OutputAttrs = ML.GetMLattrs(DataFile_path, Parameters.TrainData,Parameters.OutputArray)
        LabelNames = OutputAttrs.get('Parameters',None)
    # ==========================================================================
    # Model summary
    TrainNb,TestNb = TrainIn.shape[0],TestIn.shape[0]
    NbInput,NbOutput = TrainIn.shape[1],TrainOut.shape[1]

    ML.ModelSummary(NbInput,NbOutput,TrainNb,TestNb,FeatureNames,LabelNames)

    # ==========================================================================
    # Scale data
    # Scale input to [0,1] (based on parameter space)
    PS_bounds = np.array(Parameters.ParameterSpace).T
    InputScaler = ML.ScaleValues(PS_bounds)
    TrainIn_scale = ML.DataScale(TrainIn,*InputScaler)
    TestIn_scale = ML.DataScale(TestIn,*InputScaler)
    # Scale output to [0,1] (based on data)
    OutputScaler = ML.ScaleValues(TrainOut)
    TrainOut_scale = ML.DataScale(TrainOut,*OutputScaler)
    TestOut_scale = ML.DataScale(TestOut,*OutputScaler)
    # Convert to tensors
    TrainIn_scale = torch.from_numpy(TrainIn_scale)
    TrainOut_scale = torch.from_numpy(TrainOut_scale)
    TestIn_scale = torch.from_numpy(TestIn_scale)
    TestOut_scale = torch.from_numpy(TestOut_scale)

    # ==========================================================================
    # Train a new model or load an old one
    ModelFile = '{}/Model.pth'.format(DADict["CALC_DIR"]) # Saved model location
    if Parameters.Train:
        # get model & likelihoods
        min_noise = getattr(Parameters,'MinNoise',None)
        prev_state = getattr(Parameters,'PrevState',None)
        if prev_state==True: prev_state = ModelFile
        likelihood, model = ML.Create_GPR(TrainIn_scale, TrainOut_scale, Parameters.Kernel,
                                          prev_state=prev_state, min_noise=min_noise)

        # Train model
        TrainDict = getattr(Parameters,'TrainDict',{})
        Conv = ML.GPR_Train(model, **TrainDict)

        # Save model
        torch.save(model.state_dict(), ModelFile)

        # Plot convergence & save
        plt.figure()
        for j, _Conv in enumerate(Conv):
            plt.plot(_Conv,label='Output_{}'.format(j))
        plt.legend()
        plt.savefig("{}/Convergence.eps".format(DADict["CALC_DIR"]),dpi=600)
        plt.close()
    else:
        # Load previously trained model
        likelihood, model = ML.Create_GPR(TrainIn_scale, TrainOut_scale,
                                        Parameters.Kernel, prev_state=ModelFile)
    model.eval(); likelihood.eval()

    # =========================================================================
    # Get error metrics for model
    with torch.no_grad():
        train_pred = model(*[TrainIn_scale]*NbOutput)
        test_pred = model(*[TestIn_scale]*NbOutput)
    train_pred = np.transpose([p.mean.numpy() for p in train_pred])
    test_pred = np.transpose([p.mean.numpy() for p in test_pred])

    df_train = ML.GetMetrics2(train_pred,TrainOut_scale.detach().numpy())
    df_test = ML.GetMetrics2(test_pred,TestOut_scale.detach().numpy())
    print('\nTrain metrics')
    print(df_train)
    print('\nTest metrics')
    print(df_test,'\n')

    # ==========================================================================
    # Solve the inverse problem. Discover the combination of inputs which
    # deliver the temperatures at the thermocouple locations.

    NbCases = 5 # number of tescases to investigate
    NbInit = 100 #number of initial points for inverse solution
    confidence = [0.2,0.2,0.2,0.05,0.05,0.05,0.1,None,None]
    confidence = [[0.1]]*6+[0,None,None]
    fix_input_ix = [7]

    # fix_input_ix = list(range(7))

    true_inputs = TestIn_scale.detach().numpy()[:NbCases,:]
    target_outputs = TestOut_scale.detach().numpy()[:NbCases,:]

    # Test which shows the inverse results
    if True:
        ix=0
        args = [target_outputs[ix],model.models,fix_input_ix]

        bounds = confidence_bound(true_inputs[ix],confidence)
        init_points = init_slsqp(NbInit,bounds)


        inv_sol = InverseSolution(obj_fixed,init_points,bounds,args=args)
        with torch.no_grad():
            _inv_sol = torch.from_numpy(inv_sol[:5])
            _true_Input = torch.from_numpy(true_inputs[ix:ix+1])
            out = np.array([mod(_inv_sol).mean.numpy() for mod in model.models]).T
            out_true = np.array([mod(_true_Input).mean.numpy() for mod in model.models]).T

        print('##############################################')
        mess = "True Inputs:\n{}\nTarget Outputs:\n{}\n".format(true_inputs[ix],target_outputs[ix])
        print(mess)
        mess = "Inverse Inputs:\n{}\nPredicted Outputs:\n{}\n".format(inv_sol[:5],out)
        print(mess)

        print(out_true)
        print(((out_true - target_outputs[ix])**2).mean())


        for i in range(NbInput):
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_title(Parameters.InputParameters[i])
            ax.scatter(init_points[:,i],[0]*len(init_points),marker='x',label='Initial points')
            ax.scatter(inv_sol[:,i],[1]*len(inv_sol),marker='x',label='Inverse solution')
            ax.plot([true_inputs[ix,i]]*2, [0,1], linestyle='--',label='True solution')
            ax.set_xlim([0,1]);ax.set_ylim([-0.5,1.5])
            ax.get_yaxis().set_visible(False)
            ax.legend()
            plt.show()

    return

    err_sq_all = []
    for true_input,target_output in zip(true_inputs,target_outputs):
        args = [target_output,model.models,fix_input_ix]
        bounds = confidence_bound(true_input,confidence)
        init_points = init_slsqp(NbInit,bounds)
        inverse_sol = InverseSolution(obj_fixed,init_points,bounds,args=args)
        err_sq = np.mean((inverse_sol - true_input)**2,axis=0)
        err_sq_all.append(err_sq)
    err_sq_avg = np.array(err_sq_all).mean(axis=0)
    print(err_sq_avg)


# ==============================================================================
# Create model mapping inputs to nodal temperatures on a surface.
# This is used to predict thermocouple temperatures at variable location
def Variable_TC(VL,DADict):
    # np.random.seed(100)
    Parameters = DADict['Parameters']

    NbTorchThread = getattr(Parameters,'NbTorchThread',None)
    if NbTorchThread: torch.set_num_threads(NbTorchThread)

    if getattr(Parameters,'CompileData',None):
        CompileData(VL,DADict)

    # ==========================================================================
    # Get Train & test data from file DataFile_path
    DataFile_path = "{}/{}".format(VL.PROJECT_DIR, Parameters.DataFile)
    InputArray = getattr(Parameters,'InputArray','Input')
    OutputArray = getattr(Parameters,'OutputArray','Output')

    TrainIn, TrainOut = ML.GetMLdata(DataFile_path, Parameters.TrainData,
                                     Parameters.InputArray, Parameters.OutputArray,
                                     getattr(Parameters,'TrainNb',-1))
    TestIn, TestOut = ML.GetMLdata(DataFile_path, Parameters.TestData,
                                   Parameters.InputArray, Parameters.OutputArray,
                                   getattr(Parameters,'TestNb',-1))

    # ==========================================================================
    # Scale data
    # Scale input to [0,1] (based on parameter space)
    PS_bounds = np.array(Parameters.ParameterSpace).T
    InputScaler = ML.ScaleValues(PS_bounds)
    TrainIn_scale = ML.DataScale(TrainIn,*InputScaler)
    TestIn_scale = ML.DataScale(TestIn,*InputScaler)
    # Scale output to [0,1] (based on data)
    OutputScaler = ML.ScaleValues(TrainOut)
    TrainOut_scale = ML.DataScale(TrainOut,*OutputScaler)
    TestOut_scale = ML.DataScale(TestOut,*OutputScaler)

    # ==========================================================================
    # Compress data using svd decomposition
    U,s,VT = np.linalg.svd(TrainOut_scale,full_matrices=True)

    threshold = getattr(Parameters,'Threshold', 0.99)
    s_sc = np.cumsum(s)
    s_sc = s_sc/s_sc[-1]
    ix = np.argmax( s_sc > threshold) + 1
    VT = VT[:ix,:]
    print("PCA: Compressed {} to {} dimensions ({}% information retained)".format(len(s),ix,100*s_sc[ix]))

    # Compress Train & Test outputs
    TrainOut_scale_PCA = TrainOut_scale.dot(VT.T)
    TestOut_scale_PCA = TestOut_scale.dot(VT.T)

    TrainIn_scale = torch.from_numpy(TrainIn_scale)
    TrainOut_scale_PCA = torch.from_numpy(TrainOut_scale_PCA)
    TestIn_scale = torch.from_numpy(TestIn_scale)
    TestOut_scale_PCA = torch.from_numpy(TestOut_scale_PCA)

    # ==========================================================================
    # Model summary
    TrainNb,TestNb = TrainIn.shape[0],TestIn.shape[0]
    NbInput,NbOutput = TrainIn.shape[1],TrainOut_scale_PCA.shape[1]

    ModelDesc = "Nb.Inputs: {}\nNb.Outputs: {}\n"\
                "Nb.Train data: {}\nNb.Test data: {}\nInputs: {}\n"\
                "Outputs: {}\n".format(NbInput,NbOutput,TrainNb,TestNb,
                ", ".join(Parameters.InputParameters),
                ", ".join(Parameters.OutputParameters))
    print(ModelDesc)

    # ==========================================================================
    # Train a new model or load an old one
    ModelFile = '{}/Model.pth'.format(DADict["CALC_DIR"]) # Saved model location
    if Parameters.Train:
        # get model & likelihoods
        min_noise = getattr(Parameters,'MinNoise',None)
        prev_state = getattr(Parameters,'PrevState',None)
        if prev_state==True: prev_state = ModelFile
        likelihood, model = ML.Create_GPR(TrainIn_scale, TrainOut_scale_PCA, Parameters.Kernel,
                                          prev_state=prev_state, min_noise=min_noise)

        # Train model
        TrainDict = getattr(Parameters,'TrainDict',{})
        Conv = ML.GPR_Train(model, **TrainDict)

        # Save model
        torch.save(model.state_dict(), ModelFile)

        # Plot convergence & save
        plt.figure()
        for j, _Conv in enumerate(Conv):
            plt.plot(_Conv,label='Output_{}'.format(j))
        plt.legend()
        plt.savefig("{}/Convergence.eps".format(DADict["CALC_DIR"]),dpi=600)
        plt.close()
    else:
        # Load previously trained model
        likelihood, model = ML.Create_GPR(TrainIn_scale, TrainOut_scale_PCA,
                                        Parameters.Kernel, prev_state=ModelFile)
    model.eval(); likelihood.eval()

    # =========================================================================
    # Get error metrics for model
    with torch.no_grad():
        train_pred = model(*[TrainIn_scale]*NbOutput)
        test_pred = model(*[TestIn_scale]*NbOutput)
    train_pred = np.transpose([p.mean.numpy() for p in train_pred])
    test_pred = np.transpose([p.mean.numpy() for p in test_pred])

    df_train_PCA = ML.GetMetrics2(train_pred,TrainOut_scale_PCA.detach().numpy())
    df_test_PCA = ML.GetMetrics2(test_pred,TestOut_scale_PCA.detach().numpy())
    print('\nTrain metrics (compressed)')
    print(df_train_PCA)
    print('\nTest metrics (compressed)')
    print(df_test_PCA,'\n')

    df_train = ML.GetMetrics2(train_pred.dot(VT),TrainOut_scale)
    df_test = ML.GetMetrics2(test_pred.dot(VT),TestOut_scale)
    print('Train metrics (averaged)')
    print(df_train.mean())
    print('\nTest metrics (averaged)')
    print(df_test.mean(),'\n')

# ==============================================================================
# Optimises location for thermocouple placement used 'Variable_TC' models
def Optimise_TC(VL,DADict):
    # np.random.seed(100)
    Parameters = DADict["Parameters"]

    meshfile = "{}/{}".format(VL.MESH_DIR,Parameters.MeshFile) # mesh used in simulations

    # ==========================================================================
    # Get models
    mod_dict = {}
    for surface, dir in zip(Parameters.CandidateSurfaces,Parameters.ModelDirs):
        model_dir = "{}/{}".format(VL.PROJECT_DIR,dir)
        if not os.path.isdir(model_dir):
            sys.exit("Surface {} not avaiable".format(surface))

        mod_parameters = VLF.ReadParameters("{}/Parameters.py".format(model_dir))
        DataFile_path = "{}/{}".format(VL.PROJECT_DIR, mod_parameters.DataFile)

        TrainIn, TrainOut = ML.GetMLdata(DataFile_path, mod_parameters.TrainData,
                                         mod_parameters.InputArray, mod_parameters.OutputArray,
                                         getattr(mod_parameters,'TrainNb',-1))
        TestIn, TestOut = ML.GetMLdata(DataFile_path, 'Test',
                                       mod_parameters.InputArray, mod_parameters.OutputArray,
                                       Parameters.NbTestCases)

        PS_bounds = np.array(mod_parameters.ParameterSpace).T
        InputScaler = ML.ScaleValues(PS_bounds)
        TrainIn_scale = ML.DataScale(TrainIn,*InputScaler)
        TestIn_scale = ML.DataScale(TestIn,*InputScaler)
        OutputScaler = ML.ScaleValues(TrainOut)
        TrainOut_scale = ML.DataScale(TrainOut,*OutputScaler)
        TestOut_scale = ML.DataScale(TestOut,*OutputScaler)

        threshold = getattr(mod_parameters,'Threshold', 0.99)
        U,s,VT = np.linalg.svd(TrainOut_scale,full_matrices=True)
        s_sc = np.cumsum(s)
        s_sc = s_sc/s_sc[-1]
        ix = np.argmax(s_sc>threshold) + 1
        VT = VT[:ix,:]
        TrainOut_scale_PCA = TrainOut_scale.dot(VT.T)

        TrainIn_scale = torch.from_numpy(TrainIn_scale)
        TrainOut_scale_PCA = torch.from_numpy(TrainOut_scale_PCA)

        likelihood, model = ML.Create_GPR(TrainIn_scale, TrainOut_scale_PCA,
                                        mod_parameters.Kernel,
                                        prev_state="{}/Model.pth".format(model_dir))

        model.eval()
        model.VT = VT # attach matrix to rescale data compression
        model.test_output = TestOut_scale # Needed to find values at new points
        model.test_input = TestIn_scale
        model.surface_name = surface

        mod_dict[surface] = model

    models = [mod_dict[surf] for surf in Parameters.CandidateSurfaces]
    # confidence places a bound near the true answer depending on how confident we are
    NbInverseInit = getattr(Parameters,'NbInverseInit',100)
    GA_func = fitness_function_arg(models, meshfile, NbInverseInit, Parameters.Confidence)

    TC_space = [range(len(Parameters.CandidateSurfaces)), # discrete number for surface numbering
                {'low':0,'high':1}, # surface x1. Coordinate is scaled to [0,1] range
                {'low':0,'high':1}] # surface x2

    # Get parallelised implementation of genetic algorithm
    NbCore = getattr(Parameters,'NbCore',1)
    GA = GA_Parallel('process',NbCore)

    NbMating = getattr(Parameters,'NbMating',2)
    ga_instance = GA(num_generations=Parameters.NbGeneration,
                     num_parents_mating=NbMating,
                     gene_space=TC_space*Parameters.NbTC,
                     sol_per_pop=Parameters.NbPopulation,
                     num_genes=Parameters.NbTC*3,
                     mutation_percent_genes=100,
                     fitness_func=GA_func,
                     on_fitness=update

                     )
    ga_instance.run()
    ga_instance.plot_fitness()

    # Returning the details of the best solution.
    solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)

    print("\nOptimal thermocouple configuration\n")
    for i in range(Parameters.NbTC):
        surf_ix = int(solution[i*3])
        surf_name = Parameters.CandidateSurfaces[surf_ix]
        x1,x2 = solution[i*3+1], solution[i*3+2]
        s = "Thermocouple #{}:\n"\
            "Surface: {}\nLocation: ({}, {})\n".format(i+1,surf_name,x1,x2)
        print(s)

    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

def fitness_function_arg(surface_models, meshfile, NbInit, confidence):
    def fitness_function(solution, solution_idx):
        ModIx = solution[::3].astype('int')
        X1,X2 = solution[1::3],solution[2::3]

        MeshParameters = VLF.ReadParameters("{}.py".format(os.path.splitext(meshfile)[0]))
        MeshFile = import_module("Mesh.{}".format(MeshParameters.File))
        SurfaceNormals = MeshFile.SurfaceNormals

        TC_interp,TC_targets = [],[]
        for ix,x1,x2 in zip(ModIx,X1,X2):
            model = surface_models[ix]

            norm = SurfaceNormals[SurfaceNormals[:,0]==model.surface_name,1]
            if norm == 'NX': get = [1,2]
            elif norm == 'NY': get = [0,2]
            elif norm == 'NZ': get = [0,1]

            meshdata = MEDtools.MeshInfo(meshfile)
            group = meshdata.GroupInfo(model.surface_name)
            Coords = meshdata.GetNodeXYZ(group.Nodes)
            Coords = Coords[:,get]
            # scale coordinates to [0,1] range
            cd_min, cd_max = Coords.min(axis=0),Coords.max(axis=0)
            Coords = (Coords - cd_min)/(cd_max - cd_min)

            # Find nodes & weights to interpolate value at x1,x2
            nodes,weights = VLF.Interp_2D(Coords,group.Connect,(x1,x2))
            Interp_Ix = np.searchsorted(group.Nodes,nodes) # get index of nodes

            # Needed to make a prediction using the model
            TC_interp.append([model,Interp_Ix,weights]) # args for inverse_obj

            # ======================================================================
            # Calculate true value at thermocouple location using model
            TC_target = (model.test_output[:,Interp_Ix]*weights).sum(axis=1)
            TC_targets.append(TC_target)
        TC_targets = np.array(TC_targets).T

        # ======================================================================
        N_cases = len(TC_targets)
        torch.set_num_threads(1)
        err_sq_all = []
        for i in range(N_cases):
            true_input = model.test_input[i]
            bounds = confidence_bound(true_input,confidence)
            init_points = init_slsqp(NbInit,bounds)
            inverse_sol = InverseSolution(obj_variable, init_points, bounds,
                                          args=[TC_targets[i],TC_interp])
            err_sq = np.mean((inverse_sol - true_input)**2,axis=0)
            err_sq_all.append(err_sq)

        # average err_sq for each component & sum for single score
        err_sq_avg = np.array(err_sq_all).mean(axis=0)
        score = err_sq_avg.sum()
        print(score)

        return 1/score # return reciprocal as this is being maximised

    return fitness_function

def update(ga_instance,population_fitness):
    num_gen = ga_instance.generations_completed
    gen_best = max(population_fitness)
    best = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]
    print("Generation: {}, Gen. Best: {:.4f}, Best: {:.4f}".format(num_gen,gen_best,best))

# ==============================================================================
def confidence_bound(expected,confidence,low=0,high=1):
    # Function for creating boundary for inverse problems
    bounds = []
    for i,conf in enumerate(confidence):
        if conf==None:
            _low,_high = low,high
        else:
            _low= max(expected[i] - conf,low)
            _high = min(expected[i] + conf,high)
        bounds.append([_low,_high])
    return bounds

def init_slsqp(Nb,bounds,dist='uniform'):
    # Get initial points in domain for slsqp optimiser
    ''' Get array of initial points for slsqp optimisation'''
    points = []
    for low, high in bounds:
        if dist.lower()=='uniform':
            _points = np.random.uniform(low,high,Nb)
        points.append(_points)
    return np.array(points).T

# ==============================================================================
def InverseSolution(objfn, init_points, bounds, args=[]):
    Opt_cd, Opt_val = slsqp_multi(objfn, init_points, bounds=bounds,
                                  args=args,
                                  maxiter=30, find='min', tol=0, jac=True)
    success_bl = Opt_val<=0.01**2 # squared as the score is squared
    print(Opt_val[success_bl])
    return Opt_cd[success_bl]

def obj_fixed(X, Target, models, fix=None):
    X = torch.tensor(np.atleast_2d(X),dtype=torch_dtype)

    Preds, Grads = [], []
    for i, mod in enumerate(models):
        _Grad, _Pred = mod.Gradient_mean(X)
        Preds.append(_Pred.detach().numpy())
        Grads.append(_Grad.detach().numpy())
    Preds = np.array(Preds)
    Grads = np.swapaxes(Grads,0,1)
    if fix !=None: Grads[:,:,fix] = 0

    d = np.transpose(Preds - Target[:,None])
    Score = (d**2).mean(axis=1)
    dScore = 2*(Grads*d[:,:,None]).mean(axis=1)

    return Score, dScore

def obj_variable(X, Target, TC_interp, fix=None):
    ''' Objective function for finding the inverse solution using the slsqp
        optimisation. Minimse the sum squared error between the true values and
        predicted values at thermocouple locations.'''

    X = torch.tensor(np.atleast_2d(X),dtype=torch_dtype)

    TC_vals,TC_grads = [],[]
    for model, Ix, weights in TC_interp:
        # ======================================================================
        # Calculate predicted value & gradient at thermocouple location using model
        Preds, Grads = [], []
        for i, mod in enumerate(model.models):
            _Grad, _Pred = mod.Gradient_mean(X)
            Preds.append(_Pred.detach().numpy())
            Grads.append(_Grad.detach().numpy())
        Preds, Grads = np.array(Preds).T, np.array(Grads).T
        #Rescale from compresses data
        Preds = Preds.dot(model.VT[:,Ix])
        Grads = Grads.dot(model.VT[:,Ix])

        TC_val = (Preds*weights).sum(axis=1)
        TC_grad = (Grads*weights).sum(axis=2)
        TC_vals.append(TC_val); TC_grads.append(TC_grad.T)

    TC_vals = np.array(TC_vals).T
    TC_grads = np.swapaxes(TC_grads,0,1)

    d = (TC_vals - Target)
    Score = (d**2).sum(axis=1)
    dScore = 2*(TC_grads*d[:,:,None]).sum(axis=1)
    return Score, dScore

# ==============================================================================
# Data collection functions

def _FieldTemperatures(ResDir, InputVariables, ResFileName, ResName='Temperature'):

    # Get temperature values from results
    paramfile = "{}/Parameters.py".format(ResDir)
    Parameters = VLF.ReadParameters(paramfile)
    In = ML.GetInputs(Parameters,InputVariables)

    ResFilePath = "{}/{}".format(ResDir,ResFileName)
    Out = MEDtools.FieldResult(ResFilePath,ResName)

    return In, Out

def _SurfaceTemperatures(ResDir, SurfaceName, InputVariables,
                         ResFileName, ResName='Temperature'):
    # Get temperature values on surface 'SurfaceName'
    paramfile = "{}/Parameters.py".format(ResDir)
    Parameters = VLF.ReadParameters(paramfile)
    In = ML.GetInputs(Parameters,InputVariables)
    ResFilePath = "{}/{}".format(ResDir,ResFileName)
    Out = MEDtools.FieldResult(ResFilePath, ResName, GroupName=SurfaceName)
    return In, Out

def Get_Interp(MeshFile,SurfaceName,x1,x2,Results):
    MeshParameters = VLF.ReadParameters("{}.py".format(os.path.splitext(MeshFile)[0]))
    Mesh_File = import_module("Mesh.{}".format(MeshParameters.File))
    SurfaceNormals = Mesh_File.SurfaceNormals

    norm = SurfaceNormals[SurfaceNormals[:,0]==SurfaceName,1]
    if norm == 'NX': get = [1,2]
    elif norm == 'NY': get = [0,2]
    elif norm == 'NZ': get = [0,1]

    meshdata = MEDtools.MeshInfo(MeshFile)
    group = meshdata.GroupInfo(SurfaceName)
    Coords = meshdata.GetNodeXYZ(group.Nodes)
    Coords = Coords[:,get]
    # scale coordinates to [0,1] range
    cd_min, cd_max = Coords.min(axis=0),Coords.max(axis=0)
    Coords = (Coords - cd_min)/(cd_max - cd_min)

    # Find nodes & weights to interpolate value at x1,x2
    nodes,weights = VLF.Interp_2D(Coords,group.Connect,(x1,x2))
    Interp_Ix = np.searchsorted(group.Nodes,nodes)

    meshdata.Close()

    Interpolation = (Results[:,Interp_Ix]*weights).sum(axis=1)

    return Interpolation
