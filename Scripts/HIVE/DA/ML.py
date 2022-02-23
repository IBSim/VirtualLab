import os
import numpy as np
import torch
import gpytorch
import h5py
from natsort import natsorted
from scipy.stats import norm
import time

from Optimise import FuncOpt
from GeneticAlgorithm import ga

class ExactGPmodel(gpytorch.models.ExactGP):
    '''
    Gaussian process regression model.
    '''
    def __init__(self, train_x, train_y, likelihood, kernel,options={},ard=True):
        super(ExactGPmodel, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()

        ard_num_dims = train_x.shape[1] if ard else None
        if kernel.lower() in ('rbf'):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims))
        if kernel.lower().startswith('matern'):
            split = kernel.split('_')
            nu = float(split[1]) if len(split)==2 else 2.5
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=nu,ard_num_dims=ard_num_dims))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def Training(self,LH,Iterations=1000, lr=0.01,test=None, Print=50, ConvCheck=50,**kwargs):

        self.train()
        LH.train()
        # Use the adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)  # Includes GaussianLikelihood parameters
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(LH,self)

        ConvAvg = float('inf')
        Convergence = []
        TrainMSE, TestMSE = [],[]

        for i in range(Iterations):
            optimizer.zero_grad() # Zero gradients from previous iteration
            # Output from model
            with gpytorch.settings.max_cholesky_size(1500):
                output = self(self.train_inputs[0])
                # Calc loss and backprop gradients
                loss = -mll(output, self.train_targets)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                if i==0 or (i+1) % Print == 0:
                    out = "Iteration: {}, Loss: {}".format(i+1,loss.numpy())
                    if True:
                        out+="\nLengthscale: {}\nOutput Scale: {}\n"\
                        "Noise: {}\n".format(self.covar_module.base_kernel.lengthscale.numpy()[0],
                                             self.covar_module.outputscale.numpy(),
                                             self.likelihood.noise.numpy()[0])

                    print(out)
                    Convergence.append(loss.item())

            if test and i%50==0:
                with torch.no_grad(), gpytorch.settings.max_cholesky_size(1500),gpytorch.settings.fast_pred_var():
                    self.eval(); LH.eval()
                    x,y = test
                    pred = LH(self(x))
                    MSE = np.mean(((pred.mean-y).numpy())**2)
                    # print("MSE",MSE)
                    TestMSE.append(MSE)
                    self.train();LH.train()
                    # if (i+1) % ConvCheck == 0:
                    #     print(MSE)

            if (i+1) % ConvCheck == 0:
                Avg = np.mean(Convergence[-ConvCheck:])
                if Avg > ConvAvg:
                    print("Training terminated due to convergence")
                    break
                ConvAvg = Avg


        self.eval()
        LH.eval()
        return Convergence,TestMSE

    def Gradient(self, x):
        x.requires_grad=True
        with gpytorch.settings.fast_pred_var():
            # pred = self.likelihood(self(x))
            pred = self(x)
            grads = torch.autograd.grad(pred.mean.sum(), x)[0]
            return grads

    def Gradient_mean(self, x):
        x.requires_grad=True
        # pred = self.likelihood(self(x))
        mean = self(x).mean
        dmean = torch.autograd.grad(mean.sum(), x)[0]
        return dmean, mean

    def Gradient_variance(self, x):
        x.requires_grad=True
        with gpytorch.settings.fast_pred_var():
            # pred = self.likelihood(self(x))
            var = self(x).variance
            dvar = torch.autograd.grad(var.sum(), x)[0]
        return dvar, var

def GPRModel_Multi(TrainIn,TrainOut,Kernel,prev_state=None,min_noise=None):
    NbModel = TrainOut.shape[1]
    models,likelihoods = [], []
    for i in range(NbModel):
        _kernel = Kernel if type(Kernel)==str else Kernel[i]
        _min_noise = min_noise[i] if type(min_noise) in (list,tuple) else min_noise

        likelihood, model = GPRModel_Single(TrainIn,TrainOut[:,i],_kernel,min_noise=_min_noise)
        models.append(model)
        likelihoods.append(likelihood)

    model = gpytorch.models.IndependentModelList(*models)
    likelihood = gpytorch.likelihoods.LikelihoodList(*likelihoods)
    if prev_state:
        state_dict = torch.load(prev_state)
        model.load_state_dict(state_dict)
    return likelihood, model

def GPRModel_Single(TrainIn,TrainOut,Kernel,prev_state=None,min_noise=None):

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPmodel(TrainIn, TrainOut, likelihood, Kernel)

    if not prev_state:
        if min_noise != None:
            likelihood.noise_covar.register_constraint('raw_noise',gpytorch.constraints.GreaterThan(min_noise))

        # Start noise at lower level to avoid bad optima, but ensure it isn't zero
        noise_lower = likelihood.noise_covar.raw_noise_constraint.lower_bound
        noise_init = max(5*noise_lower,1e-8)
        # noise_init = 0
        hypers = {'likelihood.noise_covar.noise': noise_init}
        model.initialize(**hypers)
    else:
        state_dict = torch.load(prev_state)
        model.load_state_dict(state_dict)

    return likelihood, model

def GPR_Train_Multi(model,Iterations, lr=0.1, test=None, Print=50, ConvCheck=50, Verbose=False):
    likelihood = model.likelihood

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Includes GaussianLikelihood parameters
    mll = gpytorch.mlls.SumMarginalLogLikelihood(likelihood, model)


    Convergence,tol = [],1e-6
    for i in range(Iterations):
        optimizer.zero_grad() # Zero gradients from previous iteration

        with gpytorch.settings.max_cholesky_size(1500):
            output = model(*model.train_inputs)
            loss = -mll(output, model.train_targets)
            loss.backward()
            optimizer.step()

            if i==0 or (i+1) % Print == 0:
                message = "Iteration: {}, Loss: {}".format(i+1,loss.item())
                if Verbose:
                    for i,mod in enumerate(model.models):
                        Lscales = mod.covar_module.base_kernel.lengthscale.detach().numpy()[0]
                        Outscale = mod.covar_module.outputscale.detach().numpy()
                        noise = mod.likelihood.noise.detach().numpy()[0]
                        message += "Model output {}:\nLength scale: {}\nOutput scale: {}\n"\
                                  "Noise: {}\n".format(i,Lscales,Outscale,noise)
                print(message)

            Convergence.append(loss.item())

            if i>2*ConvCheck and (i+1)%ConvCheck==0:
                mean_new = np.mean(Convergence[-ConvCheck:])
                mean_old = np.mean(Convergence[-2*ConvCheck:-ConvCheck])
                if mean_new>mean_old:
                    print('Terminating due to increasing loss')
                    break
                elif np.abs(mean_new-mean_old)<tol:
                    print('Terminating due to convergence')
                    break

            # if i%100==0:
            #     with torch.no_grad():
            #         model.eval();likelihood.eval()
            #         TrainMSEtmp,TestMSEtmp = [i],[i]
            #         for i,mod in enumerate(model.models):
            #             _TrainMSE = ML.MSE(mod(TrainIn_scale).mean.numpy(),TrainOut_scale[:,i].numpy())
            #             _TestMSE = ML.MSE(mod(TestIn_scale).mean.numpy(),TestOut_scale[:,i].numpy())
            #             TrainMSEtmp.append(_TrainMSE)
            #             TestMSEtmp.append(_TestMSE)
            #         TestMSE.append(TestMSEtmp)
            #         TrainMSE.append(TrainMSEtmp)
            #         model.train();likelihood.train()

    return Convergence


# ==============================================================================
# Data scaling and rescaling functions
def DataScale(data,const,scale):
    '''
    This function scales n-dim data to a specific range.
    data: N-darray or scalar
    const: N-darray or scalar
    scale: N-darray or scalar
    Examples:
     - Normalising data:
        const=mean, scale=stddev
     - [0,1] range:
        const=min, scale=max-min
    '''
    return (data - const)/scale

def DataRescale(data,const,scale):
    '''
    This function scales data back to original range.
    data: N-darray or scalar
    const: N-darray or scalar
    scale: N-darray or scalar
    '''
    return data*scale + const

# ==============================================================================
# Metrics used to asses model performance
def MSE(Predicted,Target):
    # this is not normnalised
    sqdiff = (Predicted - Target)**2
    return np.mean(sqdiff)

def MAE(Predicted,Target):
    return np.abs(Predicted - Target).mean()/(Target.max() - Target.min())

def RMSE(Predicted,Target):
    return ((Predicted - Target)**2).mean()**0.5/(Target.max() - Target.min())

def Rsq(Predicted,Target):
    mean_pred = Predicted.mean()
    divisor = ((Predicted - mean_pred)**2).sum()
    MSE_val = ((Predicted - Target)**2).sum()
    return 1-(MSE_val/divisor)

def GetMetrics(model,x,target):
    with torch.no_grad():
        pred = model(x).mean.numpy()
    mse = MSE(pred,target)
    mae = MAE(pred,target)
    rmse = RMSE(pred,target)
    rsq = Rsq(pred,target)
    return mse,mae,rmse,rsq

# ==============================================================================
# Functions used for reading & writing data
def GetResPaths(ResDir,DirOnly=True,Skip=['_']):
    ''' This returns a naturally sorted list of the directories in ResDir'''
    ResPaths = []
    for _dir in natsorted(os.listdir(ResDir)):
        if _dir.startswith(tuple(Skip)): continue
        path = "{}/{}".format(ResDir,_dir)
        if DirOnly and os.path.isdir(path):
            ResPaths.append(path)

    return ResPaths

def Writehdf(File, array, dsetpath):
    st = time.time()
    while True:
        try:
            Database = h5py.File(File,'a')
            break
        except OSError:
            if time.time() - st > 20: break

    if dsetpath in Database:
        del Database[dsetpath]
    Database.create_dataset(dsetpath,data=array)
    Database.close()

def Readhdf(Database_path,data_paths):
    if type(data_paths)==str: data_paths = [data_paths]
    data = []
    st = time.time()
    while True:
        try:
            Database = h5py.File(Database_path,'r')
            break
        except OSError:
            if time.time() - st > 20: break
    for data_path in data_paths:
        _data = Database[data_path][:]
        data.append(_data)
    Database.close()
    return data

def GetMLdata(DataFile_path,DataNames,InputName,OutputName,Nb=-1):
    if type(DataNames)==str:DataNames = [DataNames]
    N = len(DataNames)

    data_input, data_output = [],[]
    for dataname in DataNames:
        data_input.append("{}/{}".format(dataname,InputName))
        data_output.append("{}/{}".format(dataname,OutputName))

    Data = Readhdf(DataFile_path,data_input+data_output)
    In,Out = Data[:N],Data[N:]

    for i in range(N):
        _Nb = Nb[i] if type(Nb)==list else Nb
        if _Nb==-1:continue

        if type(_Nb)==int:
            In[i] = In[i][:_Nb]
            Out[i] = Out[i][:_Nb]
        if type(_Nb) in (list,tuple):
            l,u = _Nb
            In[i] = In[i][l:u]
            Out[i] = Out[i][l:u]
    In,Out = np.vstack(In),np.vstack(Out)

    return In, Out

def WriteMLdata(DataFile_path,DataNames,InputName,OutputName,InList,OutList):
    for resname, _in, _out in zip(DataNames, InList, OutList):
        InPath = "{}/{}".format(resname,InputName)
        OutPath = "{}/{}".format(resname,OutputName)
        Writehdf(DataFile_path,_in,InPath)
        Writehdf(DataFile_path,_out,OutPath)

def CompileData(ResDirs,MapFnc,args=[]):
    In,Out = [],[]
    for ResDir in ResDirs:
        ResPaths = GetResPaths(ResDir)
        _In, _Out =[] ,[]
        for ResPath in ResPaths:
            _in, _out = MapFnc(ResPath,*args)
            _In.append(_in)
            _Out.append(_out)
        In.append(_In)
        Out.append(_Out)
    return In, Out

# ==============================================================================

def LHS_Samples(bounds,NbCandidates,seed=None,iterations=1000):
    from skopt.sampler import Lhs
    lhs = Lhs(criterion="maximin", iterations=iterations)
    Candidates = lhs.generate(bounds, NbCandidates,seed)
    return Candidates

def NN_Ix(NewPoints,OldPoints):
    # Returns the index of the nearest neighbour to NewPoitns in OldPoints
    Ixs = []
    for c in NewPoints:
        d_mag = np.linalg.norm(OldPoints - c,axis=1)
        Ixs.append(np.argmin(d_mag))
    return np.array(Ixs)

# ==============================================================================
# Adaptive scheme (no optimisation used)
def Adaptive(Candidates, model, scheme, scoring='sum',sort=True):
    score = _Adaptive(Candidates,model,scheme,scoring)

    # ==========================================================================
    # Sort
    if sort:
        # sort by sum if score is not a 1d array
        if score.ndim>1:
            sortix = np.argsort(score.sum(axis=0))[::-1]
            score, Candidates = score.T[sortix],Candidates[sortix]
        else:
            sortix = np.argsort(score)[::-1]
            score, Candidates = score[sortix],Candidates[sortix]
    return score, Candidates

# ==============================================================================
# Optimisaion using genetic algorithm
def AdaptGA(model, scheme, bounds, n_pop=100,n_gen=100, scoring='sum',sort=True):
    find='max'
    args = (model,scheme)
    coord,score = ga(_Adaptive,bounds,n_gen,n_pop,find=find,args=args)
    return score,coord

def _Adaptive(Candidates, model, scheme, scoring='sum'):
    _Candidates = torch.tensor(Candidates)
    args = [_Candidates, model]
    if scheme.lower()=='mmse':
        score = _Caller(MMSE,*args)
    elif scheme.lower()=='ei':
        score = _Caller(EI,*args)
    elif scheme.lower()=='eigf':
        score = _Caller(EIGF,*args)
    # elif scheme.lower()=='eigrad':
    #     score = _Caller(EIGrad,*args)
    elif scheme.lower()=='masa':
        score = MASA(*args)
    elif scheme.lower()=='qbc_var':
        score = QBC_Var(*args)


    # ==========================================================================
    # Combine scores
    if score.ndim>1 and scoring=='sum':
        score = score.sum(axis=0)

    return score

def _Caller(fn,Candidates,model):
    if hasattr(model,'models'):
        # Multioutput model
        score = []
        for mod in model.models:
            _score = fn(Candidates, mod)
            score.append(_score)
    else:
        # single output
        score = fn(Candidates, model)

    return np.array(score)

# ==============================================================================
# Adaptive routines implemented for stationary & genetic algorithm optimisation
def MMSE(Candidates, model):
    with torch.no_grad():
        variance = model(Candidates).variance.numpy()
    return variance

def EI(Candidates, model):
    with torch.no_grad():
        output = model(Candidates)
        pred = output.mean.numpy()
        stddev = output.stddev.numpy()

    ymin = model.train_targets.numpy().min()
    diff = ymin - pred
    z = diff/stddev
    return diff*norm.cdf(z) + stddev*norm.pdf(z)

def EIGF(Candidates, model):
    with torch.no_grad():
        output = model(Candidates)
        pred = output.mean.numpy()
        variance = output.variance.numpy()
    # ==========================================================================
    # Get nearest neighbour values (assumes same inputs for all dimensions)
    TrainIn = model.train_inputs[0].numpy()
    TrainOut = model.train_targets.numpy()
    Ixs = NN_Ix(Candidates.detach().numpy(),TrainIn)
    NN_val = TrainOut[Ixs]

    return (pred - NN_val)**2 + variance

# def EIGrad(Candidates, model):
#
#     with torch.no_grad():
#         variance = mod(Candidates).variance
#          _dmean, _mean = mod.Gradient_mean(Candidates)
#          dmean.append(_dmean.detach().numpy())
#          var.append(_var.detach().numpy())
#     dmean,var = np.array(dmean),np.array(var)
#
#     # ==========================================================================
#     # Get nearest neighbour values (assumes same inputs for all dimensions)
#     TrainIn = model.train_inputs[0][0].numpy()
#     TrainOut = np.transpose([model.train_targets[i].numpy() for i in range(NbOutput)])
#     Ixs = NN_Ix(Candidates.detach().numpy(),TrainIn)
#     NN = TrainIn[Ixs]
#     distance = NN - Candidates.detach().numpy()
#     gradsc = (distance.T[:,:,None]*dmean.T)**2
#     gradsc = gradsc.sum(axis=0).T
#     score_multi = var + gradsc
#     # print(var.T)
#     # print(gradsc.T)
#     return score_multi

def MASA(Candidates, committee):
    NbOutput = len(committee[0].models) if hasattr(committee[0],'models') else 1
    if NbOutput==1:
        d,cv = _MASA(Candidates,committee)
    else:
        d,cv = [],[]
        for i in range(NbOutput):
            _committee = [model.models[i] for model in committee]
            _d,_cv = _MASA(Candidates,_committee)
            d.append(_d);cv.append(_cv)
    d,cv = np.array(d),np.array(cv)

    return d/d.max() + cv/cv.max()

def _MASA(Candidates,committee):
    preds = []
    for model in committee:
        with torch.no_grad():
            pred = model(Candidates).mean.numpy()
        preds.append(pred)
    preds = np.transpose(preds)

    # preds lst is NbCandidate x NbCommittee
    pred_mean = preds.mean(axis=1)
    committee_sq = (preds - pred_mean[:,None])**2
    committee_var = committee_sq.mean(axis=1)

    TrainIn = committee[0].train_inputs[0].numpy()
    Ixs = NN_Ix(Candidates.detach().numpy(),TrainIn)
    NN = TrainIn[Ixs]
    distance = np.linalg.norm(NN - Candidates.detach().numpy(),axis=1)

    return distance, committee_var

def QBC_Var(Candidates, committee):
    NbOutput = len(committee[0].models) if hasattr(committee[0],'models') else 1
    if NbOutput==1:
        var,cv = _QBC_Var(Candidates,committee)
    else:
        var,cv = [],[]
        for i in range(NbOutput):
            _committee = [model.models[i] for model in committee]
            _var,_cv = _QBC_Var(Candidates,_committee)
            var.append(_var);cv.append(_cv)
    var,cv = np.array(var),np.array(cv)

    return var/var.max() + cv/cv.max()

def _QBC_Var(Candidates,committee,varavg='average'):
    preds,vars = [],[]
    for model in committee:
        with torch.no_grad():
            output = model(Candidates)
            pred = output.mean.numpy()
            variance = output.variance.numpy()
        preds.append(pred);vars.append(variance)
    preds,vars = np.transpose(preds),np.transpose(vars)

    # preds lst is NbCandidate x NbCommittee
    pred_mean = preds.mean(axis=1)
    committee_sq = (preds - pred_mean[:,None])**2
    committee_var = committee_sq.mean(axis=1)

    if varavg=='single':
        # Use the first of the committee members
        vars = vars[0]
    elif type(varavg)==int:
        # Use the best model (varavg is an index)
        vars = vars[varavg]
    else:
        vars = vars.mean(axis=1)

    return vars, committee_var

# ==============================================================================
# Optimisation using slsqp
def AdaptSLSQP(Candidates, model, scheme, bounds, constraints=(), scoring='sum',sort=True,**kwargs):
    # Finds optima in parameter space usign slsqp
    args = [model]
    if scheme.lower() == 'mmse':
        fn = _Caller_slsqp
        args.insert(0,MMSE_Grad)
    if scheme.lower() == 'qbc_var':
        fn = QBC_Var_Grad
        args += [[0],[0]]

    order = 'decreasing' if sort else None
    Optima = FuncOpt(fn, Candidates, find='max', tol=None,
                     order=order, bounds=bounds, jac=True,
                     constraints=constraints, args=args,**kwargs)
    Candidates, score = Optima

    return score, Candidates

def _Caller_slsqp(Candidates,fn,model,scoring='sum'):
    _Candidates = torch.tensor(Candidates)
    if hasattr(model,'models'):
        # Multioutput model
        score, dscore = [],[]
        for mod in model.models:
            _score,_dscore = fn(_Candidates, mod)
            score.append(_score);dscore.append(_dscore)
    else:
        # single output
        score, dscore = fn(_Candidates, model)
    score,dscore = np.array(score),np.array(dscore)

    if score.ndim>1 and scoring=='sum':
        score,dscore = score.sum(axis=0),dscore.sum(axis=0)

    return score, dscore

# ==============================================================================
# Adaptive routines implemented for slsqp optimiser
def MMSE_Grad(Candidates,model):
    dvar, var = model.Gradient_variance(Candidates)
    var, dvar = var.detach().numpy(),dvar.detach().numpy()
    return var,dvar

def QBC_Var_Grad(Candidates,committee,cvmax,vmax,scoring='sum'):
    NbOutput = len(committee[0].models) if hasattr(committee[0],'models') else 1
    if NbOutput==1:
        var,cv,dvar,dcv = _QBC_Var_Grad(Candidates,committee)
    else:
        var,cv,dvar,dcv = [],[],[],[]
        for i in range(NbOutput):
            _committee = [model.models[i] for model in committee]
            _var,_cv,_dvar,_dcv = _QBC_Var_Grad(Candidates,_committee)
            var.append(_var);cv.append(_cv)
            dvar.append(_dvar);dcv.append(_dcv)
    var,cv = np.array(var),np.array(cv)
    dvar,dcv = np.array(dvar),np.array(dcv)

    _cvmax = cv.max()
    if _cvmax>cvmax[0]: cvmax[0] = _cvmax
    _vmax = var.max()
    if _vmax>vmax[0]: vmax[0] = _vmax

    score = var/vmax + cv/cvmax
    dscore = dvar/vmax + dcv/cvmax

    if score.ndim>1 and scoring=='sum':
        score,dscore = score.sum(axis=0),dscore.sum(axis=0)

    return score,dscore

def _QBC_Var_Grad(Candidates,committee,varavg='average'):
    _Candidates = torch.tensor(Candidates)
    preds,vars,dpreds,dvars = [],[],[],[]
    for model in committee:
        dpred, pred = model.Gradient_mean(_Candidates)
        dvar, var = model.Gradient_variance(_Candidates)
        pred,dpred = pred.detach().numpy(),dpred.detach().numpy()
        var,dvar = var.detach().numpy(),dvar.detach().numpy()
        preds.append(pred);dpreds.append(dpred)
        vars.append(var);dvars.append(dvar)
    preds,dpreds = np.transpose(preds),np.transpose(dpreds)
    vars,dvars = np.array(vars),np.array(dvars)

    # preds lst is NbCandidate x NbCommittee
    pred_mean = preds.mean(axis=1)
    committee_diff = preds - pred_mean[:,None]
    committee_sq = committee_diff**2
    committee_var = committee_sq.mean(axis=1)

    dpred_mean = dpreds.mean(axis=2)
    dcommittee_diff = dpreds - dpred_mean[:,:,None]
    dcommittee_sq = 2*committee_diff.T[:,:,None]*dcommittee_diff.T
    dcommittee_var = dcommittee_sq.mean(axis=0)

    if varavg=='single':
        # Use the first of the committee members
        vars,dvars = vars[0],dvars[0]
    elif type(varavg)==int:
        # Use the best model (varavg is an index)
        vars,dvars = vars[varavg],dvars[varavg]
    else:
        vars,dvars = vars.mean(axis=0),dvars.mean(axis=0)

    return vars, committee_var, dvars, dcommittee_var

# ==============================================================================
# Constraint for slsqp optimiser
def ConstrainRad(OrigPoint, rad):
    con1 = {'type': 'ineq', 'fun': _Constrain_Multi,
            'jac':_dConstrain_Multi, 'args':[OrigPoint,rad]}
    return [con1]

def _Constrain_Multi(Candidates,OrigPoint,rad):
    a = rad**2 - np.linalg.norm(Candidates - OrigPoint,axis=1)**2
    return a

def _dConstrain_Multi(Candidates,OrigPoint,rad):
    da = -2*(Candidates - OrigPoint)
    return da
