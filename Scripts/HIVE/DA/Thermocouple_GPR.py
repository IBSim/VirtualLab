import os
import sys
import shutil

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
    if Parameters.OutputFn.lower()=="fieldtemperatures":
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

def Fixed_TC(VL,DADict):
    ''' Create model mapping inputs to thermocouple temperatures at fixed points'''
    # np.random.seed(100)
    Parameters = DADict['Parameters']

    if getattr(Parameters,'CompileData',None):
        CompileData(VL,DADict)

    # ==========================================================================
    # Get Train & test data from file DataFile_path
    DataFile_path = "{}/{}".format(VL.PROJECT_DIR, Parameters.DataFile)

    TrainIn,_TrainOut = ML.GetMLdata(DataFile_path, Parameters.TrainData,
                           Parameters.InputArray, Parameters.OutputArray,
                           getattr(Parameters,'TrainNb',-1))
    TestIn,_TestOut = ML.GetMLdata(DataFile_path, Parameters.TestData,
                          Parameters.InputArray, Parameters.OutputArray,
                          getattr(Parameters,'TestNb',-1))

    InputAttrs = ML.GetMLattrs(DataFile_path, Parameters.TrainData,Parameters.InputArray)
    FeatureNames = InputAttrs.get('Parameters',None)


    # Get temperature at points using surface temps
    meshfile = "{}/SampleHIVE.med".format(VL.MESH_DIR)
    NbTC = len(Parameters.TCLocations)
    NbTrain,NbTest = TrainIn.shape[0],TestIn.shape[0]
    TrainOut,TestOut = np.zeros((NbTrain,NbTC)), np.zeros((NbTest,NbTC))
    for i,(SurfaceName,x1,x2) in enumerate(Parameters.TCLocations):
        nds, weights = Get_Interp(meshfile,SurfaceName,x1,x2)
        TrainOut[:,i] = (_TrainOut[:,nds]*weights).sum(axis=1)
        TestOut[:,i] = (_TestOut[:,nds]*weights).sum(axis=1)

    LabelNames = "Thermocouples placed at the following locations:\n{}".format(Parameters.TCLocations)


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
                                          prev_state=prev_state, min_noise=min_noise,
                                          input_scale=InputScaler,output_scale=OutputScaler)

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
                                        Parameters.Kernel, prev_state=ModelFile,
                                        input_scale=InputScaler,output_scale=OutputScaler)
    model.eval(); likelihood.eval()

    # =========================================================================
    # Get error metrics for model

    with torch.no_grad():
        train_pred = model(*[TrainIn_scale]*NbOutput)
        test_pred = model(*[TestIn_scale]*NbOutput)

    train_pred_scale = np.transpose([p.mean.numpy() for p in train_pred])
    test_pred_scale = np.transpose([p.mean.numpy() for p in test_pred])
    train_pred = ML.DataRescale(train_pred_scale,*OutputScaler)
    test_pred = ML.DataRescale(test_pred_scale,*OutputScaler)

    df_train = ML.GetMetrics2(train_pred,TrainOut)
    df_test = ML.GetMetrics2(test_pred,TestOut)
    print('\nTrain metrics')
    print(df_train)
    print('\nTest metrics')
    print(df_test,'\n')

    # ==========================================================================
    # See impact of varying inputs
    Plot1D = getattr(Parameters,'Plot1D',{})
    if Plot1D:
        base = Plot1D['Base']
        ncol = Plot1D.get('NbCol',1)

        for j, mod in enumerate(model.models):
            mean,stdev = ML.InputQuery(mod,NbInput,base=0.5)
            nrow = int(np.ceil(NbInput/ncol))
            fig,ax = plt.subplots(nrow,ncol,figsize=(15,15))
            axes = ax.flatten()

            base_in = [base]*NbInput
            with torch.no_grad():
                base_val = mod(torch.tensor([base_in])).mean.numpy()

            base_in = ML.DataRescale(np.array(base_in),*InputScaler)
            base_val = ML.DataRescale(base_val,*OutputScaler[:,j])

            for i, (val,std) in enumerate(zip(mean,stdev)):
                val = ML.DataRescale(val,*OutputScaler[:,j])
                std = ML.DataRescale(std,0,OutputScaler[1,j])
                axes[i].title.set_text(FeatureNames[i])
                _split = np.linspace(InputScaler[0,i],InputScaler[:,i].sum(),len(val))

                axes[i].plot(_split,val)
                axes[i].fill_between(_split, val-2*std, val+2*std, alpha=0.5)
                axes[i].scatter(base_in[i],base_val)

            fig.text(0.5, 0.04, 'Parameter range', ha='center')
            fig.text(0.04, 0.5, OutputLabels[j], va='center', rotation='vertical')

            plt.show()

    # ==========================================================================
    # Solve the inverse problem. Discover the combination of inputs which
    # deliver the temperatures at the thermocouple locations.

    # Test which shows the inverse results
    if hasattr(Parameters,'InverseSolution'):
        IS = Parameters.InverseSolution
        NbInit = IS.get('NbInit',100)
         # initial points for slsqp
        true_inputs = TestIn_scale.detach().numpy()
        target_outputs = TestOut
        ix = IS['Index']
        confidence = IS.get('Confidence',[1]*NbInput)
        # ixs = np.where(true_inputs[:,-1]>=0.2)[0]
        # ix = ixs[19]

        fix = np.where(np.array(confidence)==1)[0]
        args = [target_outputs[ix],model.models,fix]

        init_points, bounds = ranger(NbInit,true_inputs[ix],confidence)
        init_points = np.transpose(init_points)

        inv_sol, error = InverseSolution(obj_fixed,init_points,bounds,args=args,tol=0.05)
        with torch.no_grad():
            _inv_sol = torch.from_numpy(inv_sol[:])
            pred_inv = [mod(_inv_sol) for mod in model.models]
            mean_inv = np.array([p.mean.numpy() for p in pred_inv]).T
            stddev_inv = np.array([p.stddev.numpy() for p in pred_inv]).T
            mean_inv = ML.DataRescale(mean_inv,*OutputScaler)
            stddev_inv = ML.DataRescale(stddev_inv,0,OutputScaler[1])
            mse_inv = ((mean_inv - target_outputs[ix])**2).mean(axis=1)

            _true_Input = torch.from_numpy(true_inputs[ix:ix+1])
            pred_sol = [mod(_true_Input) for mod in model.models]
            mean_sol = np.array([p.mean.numpy() for p in pred_sol]).T
            stddev_sol = np.array([p.stddev.numpy() for p in pred_sol]).T
            mean_sol = ML.DataRescale(mean_sol,*OutputScaler)
            stddev_sol = ML.DataRescale(stddev_sol,0,OutputScaler[1])
            mse_sol = ((mean_sol - target_outputs[ix])**2).mean()

        print('##############################################\n')
        print("TC Temperatures")
        print("Target:\n{}".format(target_outputs[ix]))
        print("Best model:")
        for mn,std,mse in zip(mean_inv,stddev_inv,mse_inv):
            print("{} (Err:{}, Stddev:{})".format(mn, mse, stddev_inv.sum()))
        print("True model:\n{} (Err:{}, Sum stddev:{})".format(mean_sol,mse_sol,stddev_sol.sum()))

        print("\nInputs (Scaled)")
        print("Target:\n{}".format(true_inputs[ix]))
        print("Inverse:")
        for sol in inv_sol:
            print(sol)

        print('\n##############################################\n')

        for i in range(NbInput):
            if confidence[i]==1: continue
            # if i<=6: continue
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_title(FeatureNames[i])
            # ax.scatter(init_points[:,i],[0]*len(init_points),marker='x',label='Initial points')
            ax.scatter(inv_sol[:,i],[0]*len(inv_sol), marker='x',label='Inverse solution')
            ax.scatter([true_inputs[ix,i]], [0], marker='o',label='True solution')
            ax.set_xlim([0,1]);ax.set_ylim([-0.5,1.5])
            ax.get_yaxis().set_visible(False)
            ax.legend()
            plt.show()


# ==============================================================================

def Optimise_Field(VL,DADict):
    ''' Create model mapping inputs to nodal temperatures of the whole result field.
    This is used to optimise the thermocouple placements.'''

    # np.random.seed(100)
    Parameters = DADict['Parameters']

    NbTorchThread = getattr(Parameters,'NbTorchThread',None)
    if NbTorchThread: torch.set_num_threads(NbTorchThread)

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

    InputAttrs = ML.GetMLattrs(DataFile_path, Parameters.TrainData, Parameters.InputArray)
    FeatureNames = InputAttrs.get('Parameters',None)
    OutputAttrs = ML.GetMLattrs(DataFile_path, Parameters.TrainData,Parameters.OutputArray)
    LabelNames = OutputAttrs.get('Parameters',None)

    # ==========================================================================
    # Compress data using svd decomposition
    VT_file = "{}/VT".format(DADict['CALC_DIR'])
    if os.path.isfile("{}.npy".format(VT_file)) and not getattr(Parameters,'NewCompression',False):
        VT = np.load("{}.npy".format(VT_file))
    else:
        U,s,VT = np.linalg.svd(TrainOut,full_matrices=False)

        threshold = getattr(Parameters,'Threshold', 0.99)
        s_sc = np.cumsum(s)
        s_sc = s_sc/s_sc[-1]
        ix = np.argmax( s_sc > threshold) + 1
        VT = VT[:ix,:]
        np.save(VT_file,VT)
        print("PCA: Compressed {} to {} dimensions ({}% information retained)".format(TrainOut.shape[1],ix,100*s_sc[ix]))

    # Compress Train & Test outputs
    TrainOut_PCA = TrainOut.dot(VT.T)
    TestOut_PCA = TestOut.dot(VT.T)

    # ==========================================================================
    # Scale data
    # Scale input to [0,1] (based on parameter space)
    PS_bounds = np.array(Parameters.ParameterSpace).T
    InputScaler = ML.ScaleValues(PS_bounds)
    TrainIn_scale = ML.DataScale(TrainIn,*InputScaler)
    TestIn_scale = ML.DataScale(TestIn,*InputScaler)
    # Scale output to [0,1] (based on data)
    OutputScaler = ML.ScaleValues(TrainOut_PCA)
    TrainOut_PCA_scale = ML.DataScale(TrainOut_PCA,*OutputScaler)
    TestOut_PCA_scale = ML.DataScale(TestOut_PCA,*OutputScaler)

    TrainIn_scale = torch.from_numpy(TrainIn_scale)
    TrainOut_PCA_scale = torch.from_numpy(TrainOut_PCA_scale)
    TestIn_scale = torch.from_numpy(TestIn_scale)
    TestOut_PCA_scale = torch.from_numpy(TestOut_PCA_scale)

    # ==========================================================================
    # Model summary
    TrainNb,TestNb = TrainIn.shape[0],TestIn.shape[0]
    NbInput,NbOutput = TrainIn.shape[1],TrainOut_PCA_scale.shape[1]

    ML.ModelSummary(NbInput,NbOutput,TrainNb,TestNb,FeatureNames,LabelNames)

    # ==========================================================================
    # Train a new model or load an old one
    ModelFile = '{}/Model.pth'.format(DADict["CALC_DIR"]) # Saved model location
    if Parameters.Train:
        # get model & likelihoods
        min_noise = getattr(Parameters,'MinNoise',None)
        prev_state = getattr(Parameters,'PrevState',None)
        if prev_state==True: prev_state = ModelFile
        likelihood, model = ML.Create_GPR(TrainIn_scale, TrainOut_PCA_scale, Parameters.Kernel,
                                          prev_state=prev_state, min_noise=min_noise,
                                          input_scale=InputScaler,output_scale=OutputScaler)

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
        likelihood, model = ML.Create_GPR(TrainIn_scale, TrainOut_PCA_scale, Parameters.Kernel,
                            prev_state=ModelFile,input_scale=InputScaler,output_scale=OutputScaler)
    model.eval(); likelihood.eval()

    # =========================================================================
    # Get error metrics for model
    with torch.no_grad():
        train_pred = model(*[TrainIn_scale]*NbOutput)
        test_pred = model(*[TestIn_scale]*NbOutput)
    train_pred_PCA = np.transpose([p.mean.numpy() for p in train_pred])
    test_pred_PCA = np.transpose([p.mean.numpy() for p in test_pred])

    df_train_PCA = ML.GetMetrics2(train_pred_PCA,TrainOut_PCA_scale.detach().numpy())
    df_test_PCA = ML.GetMetrics2(test_pred_PCA,TestOut_PCA_scale.detach().numpy())
    print('\nTrain metrics (compressed)')
    print(df_train_PCA)
    print('\nTest metrics (compressed)')
    print(df_test_PCA,'\n')

    train_pred_rescale = ML.DataRescale(train_pred_PCA,*OutputScaler)
    test_pred_rescale = ML.DataRescale(test_pred_PCA,*OutputScaler)
    df_train = ML.GetMetrics2(train_pred_rescale.dot(VT),TrainOut)
    df_test = ML.GetMetrics2(test_pred_rescale.dot(VT),TestOut)
    print('Train metrics (averaged)')
    print(df_train.mean())
    print('\nTest metrics (averaged)')
    print(df_test.mean(),'\n')


    # ==========================================================================
    # Set thermocouple placement, either manually or by optimisation
    model.VT = VT
    meshfile = "{}/{}".format(VL.MESH_DIR,Parameters.MeshFile)

    if hasattr(Parameters,'Optimise'):
        # optimise the locations for the thermocouples
        Optimise = Parameters.Optimise
        CandidateSurfaces = Optimise['CandidateSurfaces']

        OptDict = {'Confidence':Optimise['Confidence'],
                   'Nbslsqp':Optimise.get('Nbslsqp',100)}
        OptDict['Target_Temp'] = TestOut[-Optimise['NbTestCases']:]
        OptDict['Target_Soln'] = TestIn_scale.detach().numpy()[-Optimise['NbTestCases']:]

        GA_func = ff_field(model, meshfile, CandidateSurfaces, OptDict)

        TC_space = [range(len(CandidateSurfaces)), # discrete number for surface numbering
                    {'low':0,'high':1},{'low':0,'high':1}] # x1,x2 coordinate is scaled to [0,1] range

        # Get parallelised implementation of genetic algorithm
        NbCore = Optimise.get('NbCore',1)
        GA = GA_Parallel('process',NbCore)
        NbMating = Optimise.get('NbMating',2)
        NbGen = Optimise['NbGeneration']
        NbTC = Optimise['NbTC']
        NbPop = Optimise['NbPopulation']

        ga_instance = GA(num_generations=NbGen,
                         num_parents_mating=NbMating,
                         gene_space=TC_space*NbTC,
                         sol_per_pop=NbPop,
                         num_genes=NbTC*3,
                         mutation_percent_genes=50,
                         fitness_func=GA_func,
                         on_fitness=update)

        ga_instance.run()

        # ======================================================================
        # plot ga performance
        plt.figure()
        plt.plot(ga_instance.best_solutions_fitness, linewidth=2, color='b')
        plt.xlabel('Generation',fontsize=14)
        plt.ylabel('Fitness',fontsize=14)
        plt.savefig("{}/GA_history.png".format(DADict['CALC_DIR']))
        plt.close()

        # ======================================================================
        # Returning the details of the best solution.
        solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)

        print("\nOptimal thermocouple configuration\n")
        TCLocations = []
        for i in range(NbTC):
            surf_ix = int(solution[i*3])
            surf_name = CandidateSurfaces[surf_ix]
            x1,x2 = solution[i*3+1], solution[i*3+2]
            s = "Thermocouple #{}:\n"\
                "Surface: {}\nLocation: ({:.4f}, {:.4f})\n".format(i+1,surf_name,x1,x2)
            print(s)
            TCLocations.append([surf_name,x1,x2])

    elif hasattr(Parameters,'TCLocations'):
        # Use the pre-set thermocouple locations
        TCLocations = Parameters.TCLocations
    else:
        print('No Thermocouple locations specified')
        return

    # ==========================================================================
    # Create images of component to show location of thermocouples
    if getattr(Parameters,'TCImages',True):
        from Scripts.Common.VLPackages.Salome import Salome
        Centres = []
        for SurfName,x1,x2 in TCLocations:
            Coords = GetCoords(meshfile,SurfName,x1,x2)
            Centres.append(Coords)
        OutputDir="{}/TCPlacements".format(DADict['CALC_DIR'])
        os.makedirs(OutputDir,exist_ok=True )
        DataDict = {'Centres':Centres,
                    'File':meshfile,
                    'OutputDir':OutputDir}

        dir_path = os.path.dirname(os.path.realpath(__file__))
        Script = "{}/PV_TCPlacement.py".format(dir_path)
        Salome.Run(Script, GUI=False, DataDict=DataDict,tempdir=VL.TEMP_DIR)

    # ==========================================================================
    # Make results file containing the true solution & inversely informed temperature fields
    Compare = getattr(Parameters,'Compare',{})
    if Compare:
        ix = Compare['Ix']
        confidence = Compare['Confidence']
        Nbslsqp = Compare.get('Nbslsqp',100)

        TC_interp = Interpolate_TC(TCLocations,meshfile)

        # True values at TC location
        TC_targets = []
        for nodes,weights in TC_interp:
            TC_target = (TestOut[ix,nodes]*weights).sum()
            TC_targets.append(TC_target)

        true_input = TestIn_scale.detach().numpy()[ix]

        fix = np.where(np.array(confidence)==1)[0]
        init_points, bounds = ranger(Nbslsqp,true_input,confidence)
        init_points = np.transpose(init_points)

        inverse_sol,error = InverseSolution(obj_field, init_points, bounds,tol=0.05,
                                      args=[TC_targets,TC_interp,model,fix])

        # Make prediction of field using inverse solution as input
        with torch.no_grad():
            _inverse_sol = torch.from_numpy(inverse_sol)
            preds = []
            for mod in model.models:
                pred = mod(_inverse_sol).mean.numpy()
                pred = ML.DataRescale(pred,*mod.output_scale)
                preds.append(pred)
            preds = np.transpose(preds)

        preds = preds.dot(model.VT) # Full temperature field

        UniquePred,Ix = preds[:1],[0]
        for i, pred in enumerate(preds[1:]):
            diff = UniquePred - pred
            diff_sc = np.abs(diff)/pred
            diff_mean = diff_sc.mean(axis=1)
            diff_max = diff_sc.max(axis=1)
            if False:
                print(i+1)
                print("Mean difference",diff.mean(axis=1))
                print("St.Dev",np.std(diff,axis=1))
                print("Scaled mean",diff_mean)
                print("Scaled_max",diff_max)
                print()

            #only keep ones which are different to the others
            if (diff_mean>0.025).all():
                UniquePred = np.vstack((UniquePred,pred))
                Ix.append(i+1)

        UniqueIS = inverse_sol[Ix]
        UniqueIS = ML.DataRescale(UniqueIS,*InputScaler)
        print(UniqueIS)

        # Make rmed results file containing inverse field & true field
        ML_resfile = "{}/ML_res.rmed".format(DADict['CALC_DIR'])
        shutil.copy2(meshfile,ML_resfile)
        ndigit = len(str(len(UniquePred)))
        for i,sol in enumerate(UniquePred):
            AddResult(ML_resfile,sol,'InverseSol_{}'.format(str(i).zfill(ndigit)))
        AddResult(ML_resfile,TestOut[ix],'TrueSol')

    # InverseDict = Parameters.InverseDict
    # NbCases = InverseDict.get('NbCases',len(TestOut))
    # InverseDict['Target_Temp'] = TestOut[:NbCases]
    # InverseDict['Target_Soln'] = TestIn_scale.detach().numpy()[:NbCases]
    #
    # score = field_inverse(TCLocations, model, meshfile,InverseDict)
    # print(score)


# ==============================================================================
# functions used by pyGAD to optimise TC placement
def ff_field(model, meshfile, CandidateSurfaces, InverseDict):
    ''' Objective function used by pygad. aims to maximise the returned value. '''
    def fitness_function(solution, solution_idx):
        # ======================================================================
        # Convert surface index to surface name
        TCData = []
        for i in range(0,len(solution),3):
            SurfName = CandidateSurfaces[int(solution[i])]
            TCData.append([SurfName,solution[i+1],solution[i+2]])

        score = field_inverse(TCData,model,meshfile,InverseDict)
        return 1/score # return reciprocal as this is being maximised

    return fitness_function

def update(ga_instance,population_fitness):
    ''' Used by pygad for updates.'''
    num_gen = ga_instance.generations_completed
    gen_best = max(population_fitness)
    best = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]
    print("Generation: {}, Gen. Best: {:.4f}, Best: {:.4f}".format(num_gen,gen_best,best))

def field_inverse(TCData, model, meshfile, InvDict):
    ''' Function which analyses how successfull a set of thermocouples are at
    finding the correct inverse solution. '''

    # ==========================================================================
    # Get interpolation information for TC location
    TC_interp = Interpolate_TC(TCData,meshfile)
    # ======================================================================
    # Calculate true value at thermocouple location
    TC_targets = []
    for nodes,weights in TC_interp:
        TC_target = (InvDict['Target_Temp'][:,nodes]*weights).sum(axis=1)
        TC_targets.append(TC_target)

    # ======================================================================
    N_cases = len(TC_targets)
    torch.set_num_threads(1)
    err_sq_all = []
    NbInit = InvDict.get('Nbslsqp',100)

    confidence = InvDict.get('Confidence',None)
    fix = np.where(np.array(confidence)==1)[0]

    for i in range(N_cases):
        true_input = InvDict['Target_Soln'][i]
        init_points, bounds = ranger(NbInit,true_input,confidence)
        init_points = np.transpose(init_points)

        inverse_sol,error = InverseSolution(obj_field, init_points, bounds,tol=0.05,
                                      args=[TC_targets[i],TC_interp,model,fix])
        inverse_sol = inverse_sol[0]

        err_sq = np.mean((inverse_sol - true_input)**2,axis=0)
        err_sq_all.append(err_sq)

    # average err_sq for each component & sum for single score
    err_sq_avg = np.array(err_sq_all).mean(axis=0)
    score = err_sq_avg.sum()

    return score

# ==============================================================================
# Function used to calculate inverse solutions & the objective functions used
def InverseSolution(objfn, init_points, bounds, args=[],tol=0):
    ''' Solve inverse problem using slsqp multi routine.'''
    Best_Input, Error = slsqp_multi(objfn, init_points, bounds=bounds,
                                  args=args,
                                  maxiter=30, find='min', tol=tol, jac=True)
    return Best_Input, Error


def obj_fixed(X, Target, models, fix=None):
    ''' Objective function for finding the inverse solution using the slsqp
    optimisation. Minimse the sum squared error between the true values and
    predicted values at fixed thermocouple locations.'''

    # Calculate values & gradients using model
    X = torch.tensor(np.atleast_2d(X),dtype=torch_dtype)
    Preds, Grads = [], []
    for i, mod in enumerate(models):
        _Grad, _Pred = mod.Gradient_mean(X)
        _Pred = ML.DataRescale(_Pred.detach().numpy(),*mod.output_scale)
        _Grad = ML.DataRescale(_Grad.detach().numpy(),0,mod.output_scale[1])
        Preds.append(_Pred); Grads.append(_Grad)

    Preds = np.array(Preds)
    Grads = np.swapaxes(Grads,0,1)
    if fix is not None: Grads[:,:,fix] = 0

    # ==========================================================================
    # Calculate error

    d = np.transpose(Preds - Target[:,None])
    Score = (d**2).mean(axis=1)
    dScore = 2*(Grads*d[:,:,None]).mean(axis=1)

    return Score, dScore

def obj_field(X, Target, TC_interp, model, fix=None):
    ''' Objective function for finding the inverse solution using the slsqp
    optimisation. Minimse the sum squared error between the true values and
    predicted values at variable thermocouple locations.'''

    # Calculate values & gradients using model
    X = torch.tensor(np.atleast_2d(X),dtype=torch_dtype)
    Preds, Grads = [], []
    for i, mod in enumerate(model.models):
        _Grad, _Pred = mod.Gradient_mean(X)
        _Pred = ML.DataRescale(_Pred.detach().numpy(),*mod.output_scale)
        _Grad = ML.DataRescale(_Grad.detach().numpy(),0,mod.output_scale[1])

        Preds.append(_Pred); Grads.append(_Grad)
    Preds, Grads = np.array(Preds).T, np.array(Grads).T

    # ==========================================================================
    # Decompress data & find values at points described by Ix & weights
    TC_vals,TC_grads = [],[]
    for Ix, weights in TC_interp:
        TC_val = (Preds.dot(model.VT[:,Ix])*weights).sum(axis=1)
        TC_grad = (Grads.dot(model.VT[:,Ix])*weights).sum(axis=2)
        TC_vals.append(TC_val); TC_grads.append(TC_grad.T)

    TC_vals = np.array(TC_vals).T
    TC_grads = np.swapaxes(TC_grads,0,1)
    # Change gradients to zero if fix provided
    if fix is not None: TC_grads[:,:,fix] = 0

    # ==========================================================================
    # Calculate error
    d = (TC_vals - Target)
    Score = (d**2).mean(axis=1)
    dScore = 2*(TC_grads*d[:,:,None]).mean(axis=1)

    return Score, dScore

# ==============================================================================
# Data collection functions
def _FieldTemperatures(ResDir, InputVariables, ResFileName, ResName='Temperature'):
    ''' Get temperature values at all nodes'''
    # Get temperature values from results
    paramfile = "{}/Parameters.py".format(ResDir)
    Parameters = VLF.ReadParameters(paramfile)
    In = ML.GetInputs(Parameters,InputVariables)

    ResFilePath = "{}/{}".format(ResDir,ResFileName)
    Out = MEDtools.FieldResult(ResFilePath,ResName)

    return In, Out

# ==============================================================================
# Useful functions
def ranger(Nb,expected_value,confidence=None,low=0,high=1):
    ''' Get initial points for slsqp optimiser & bounds for the problem.
    These are calculated using the amount of confidence we have about the
    'true' value. '''
    if confidence==None:confidence = [0]*len(expected_value)
    bounds,points = [],[]
    for i, val in enumerate(expected_value):
        if confidence[i]==0:
            _bounds = [low,high]
            _points = np.random.uniform(low,high,Nb)
        elif confidence[i]==1:
            _bounds = [val,val]
            _points = np.ones(Nb)*val
        else:
            stand_dev = 1/6 # ensures +- 0.5 from expected contains 3 stan. devs.
            scaled_std = stand_dev*(1-confidence[i])
            bnd_min = max(val-3*scaled_std,0)
            bnd_max = min(val+3*scaled_std,1)
            _bounds = [bnd_min,bnd_max]
            _points = np.random.normal(val,scaled_std,size=Nb)
            _points[_points<bnd_min] = bnd_min
            _points[_points>bnd_max] = bnd_max
        bounds.append(_bounds)
        points.append(_points)
    return points, bounds

def GetNorm(MeshFile,SurfaceName):
    ''' Get norm to surface from mesh creation file'''
    MeshParameters = VLF.ReadParameters("{}.py".format(os.path.splitext(MeshFile)[0]))
    Mesh_File = import_module("Mesh.{}".format(MeshParameters.File))
    SurfaceNormals = Mesh_File.SurfaceNormals
    norm = SurfaceNormals[SurfaceNormals[:,0]==SurfaceName,1]
    return norm

def GetCoords(MeshFile,SurfaceName,x1,x2):
    ''' Calculate the coordinate of a thermocouple when provided in the
    standard thermocouple format, i.e. (SurfaceName,x1,x2)'''
    # Get coordinates of group
    meshdata = MEDtools.MeshInfo(MeshFile)
    group = meshdata.GroupInfo(SurfaceName)
    Coords = meshdata.GetNodeXYZ(group.Nodes)
    cd_min, cd_max = Coords.min(axis=0),Coords.max(axis=0)

    # decide what coordinates to keep/change based on surface normal
    norm = GetNorm(MeshFile,SurfaceName)
    if norm == 'NX': get = [1,2]
    elif norm == 'NY': get = [0,2]
    elif norm == 'NZ': get = [0,1]
    _cd_min,_cd_max = cd_min[get],cd_max[get]

    # calculate coordinate
    NewCoord = cd_min.copy()
    NewCoord[get] = _cd_min + np.array([x1,x2])*(_cd_max - _cd_min)

    meshdata.Close()

    return NewCoord

def Interpolate_TC(TCData,meshfile):
    ''' Get nodes indexes & weights for all thermocouples provided'''
    Interp = []
    for SurfName,x1,x2 in TCData:
        nodes, weights = Get_Interp(meshfile,SurfName,x1,x2)
        Interp.append([nodes,weights])
    return Interp

def Get_Interp(MeshFile,SurfaceName,x1,x2):
    ''' Get the node index & weights to inteprolate value at a point on the
    surface of the sample for TC measurements.'''

    # Get coordinates of the group
    meshdata = MEDtools.MeshInfo(MeshFile)
    group = meshdata.GroupInfo(SurfaceName)
    Coords = meshdata.GetNodeXYZ(group.Nodes)

    # Know which coordinates to keep based on the surface normal
    norm = GetNorm(MeshFile,SurfaceName)
    if norm == 'NX': get = [1,2]
    elif norm == 'NY': get = [0,2]
    elif norm == 'NZ': get = [0,1]
    Coords = Coords[:,get]

    # scale coordinates to [0,1] range
    cd_min, cd_max = Coords.min(axis=0),Coords.max(axis=0)
    Coords = (Coords - cd_min)/(cd_max - cd_min)

    # Find nodes & weights to interpolate value at x1,x2
    nodes,weights = VLF.Interp_2D(Coords,group.Connect,(x1,x2))

    meshdata.Close()

    return nodes, weights

def AddResult(file,array,resname):
    ''' Creates codeaster style results field in a med file.
    Result 'resname' is created consisting of 'array'. '''

    h5py_file = h5py.File(file,'a')

    from Scripts.Common.tools import MEDtools
    Formats = h5py.File("{}/MED_Format.med".format(os.path.dirname(MEDtools.__file__)),'r')
    GrpFormat = Formats['ELEME']
    h5py_file.copy(GrpFormat,"CHA/{}".format(resname))
    grp = h5py_file["CHA/{}".format(resname)]
    Formats.close()

    grp.attrs.create('MAI','Sample',dtype='S8')
    if array.ndim == 1:
        NOM,NCO =  'Res'.ljust(16),1
    else:
        NOM, NCO = '', array.shape[1]
        for i in range(NCO):
            NOM+=('Res{}'.format(i)).ljust(16)

    # ==========================================================================
    # formats needed for paravis
    grp.attrs.create('NCO',NCO,dtype='i4')
    grp.attrs.create('NOM', NOM,dtype='S100')
    grp.attrs.create('TYP',6,dtype='i4')
    grp.attrs.create('UNI',''.ljust(len(NOM)),dtype='S100')
    grp.attrs.create('UNT','',dtype='S1')
    grp = grp.create_group('0000000000000000000000000000000000000000')
    grp.attrs.create('NDT',0,dtype='i4')
    grp.attrs.create('NOR',0,dtype='i4')
    grp.attrs.create('PDT',0.0,dtype='f8')
    grp.attrs.create('RDT',-1,dtype='i4')
    grp.attrs.create('ROR',-1,dtype='i4')
    grp = grp.create_group('NOE')
    grp.attrs.create('GAU','',dtype='S1')
    grp.attrs.create('PFL','MED_NO_PROFILE_INTERNAL',dtype='S100')
    grp = grp.create_group('MED_NO_PROFILE_INTERNAL')
    grp.attrs.create('GAU','',dtype='S1'    )
    grp.attrs.create('NBR', array.shape[0], dtype='i4')
    grp.attrs.create('NGA',1,dtype='i4')
    grp.create_dataset("CO",data=array.flatten(order='F'))

    h5py_file.close()
