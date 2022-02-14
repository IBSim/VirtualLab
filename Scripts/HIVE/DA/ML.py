import os
import numpy as np
import torch
import gpytorch
import h5py
import pickle
from natsort import natsorted
from importlib import import_module, reload
from scipy.stats import norm

from Optimise import FuncOpt

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

    def EI(self, Candidates, sort=True):
        with torch.no_grad():
            _Candidates = torch.tensor(Candidates)
            score = self(_Candidates).variance.numpy()

        if sort:
            sortix = np.argsort(score)[::-1]
            score, Candidates = score[sortix],Candidates[sortix]

        return score, Candidates

    def EIGF(self, Candidates, NN_val, sort=True):
        with torch.no_grad():
            _Candidates = torch.tensor(Candidates)
            comp = self(_Candidates)
            pred = comp.mean.numpy()
            UQ = comp.variance.numpy()
            score = UQ + (pred - NN_val)**2

        if sort:
            sortix = np.argsort(score)[::-1]
            score, Candidates = score[sortix],Candidates[sortix]

        return score, Candidates

    def MaximisedEI(self, Candidates, bounds, tol=0, sort=True):
        order = 'decreasing' if sort else None
        Optima = FuncOpt(_MaxUQ, Candidates, find='max', tol=None,
                         order=order, bounds=bounds, jac=True, args=[self])
        Candidates, score = Optima
        return score, Candidates

def _MaxUQ(Candidates,GPR_model):
    _Candidates = torch.tensor(Candidates)
    dvar, var = GPR_model.Gradient_variance(_Candidates)
    dvar, var = dvar.detach().numpy(), var.detach().numpy()
    return var, dvar

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

def MSE(Predicted,Target):
    sqdiff = (Predicted - Target)**2
    return np.mean(sqdiff)

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
    Database = h5py.File(File,'a')
    if dsetpath in Database:
        del Database[dsetpath]
    Database.create_dataset(dsetpath,data=array)
    Database.close()

def Readhdf(Database_path,data_paths):
    if type(data_paths)==str: data_paths = [data_paths]
    data = []
    Database = h5py.File(Database_path,'r')
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

def LHS_Samples(bounds,NbCandidates,seed=None):
    from skopt.sampler import Lhs
    lhs = Lhs(criterion="maximin", iterations=1000)
    Candidates = lhs.generate(bounds, NbCandidates,seed)
    return Candidates

def Var_Multi(model, Candidates, scoring='sum',sort=True):
    NbOutput = len(model.models)
    # ==========================================================================
    # Use model to make prediction
    with torch.no_grad():
        _Candidates = torch.tensor(Candidates)
        output = model(*[_Candidates]*NbOutput)

    # ==========================================================================
    # Calculate score & combine in to single value
    vars = [out.variance.numpy() for out in output]
    score_multi = np.transpose(vars)

    if scoring=='sum':
        score = score_multi.sum(axis=1)
    # Todo, f1 score

    # ==========================================================================
    # Sort, if required
    if sort:
        sortix = np.argsort(score)[::-1]
        score, Candidates = score[sortix],Candidates[sortix]

    return score, Candidates

def EI_Multi(model, Candidates, scoring='sum',sort=True):
    NbOutput = len(model.models)

    # ==========================================================================
    # Use model to make prediction
    with torch.no_grad():
        _Candidates = torch.tensor(Candidates)
        output = model(*[_Candidates]*NbOutput)

    # ==========================================================================
    # Calculate score & combine in to single value
    score_multi = []
    for i,out in enumerate(output):
        ymin = model.train_targets[i].numpy().min()
        stddev = out.stddev.numpy()
        diff = ymin - out.mean.numpy()
        z = diff/stddev
        score_ind = diff*norm.cdf(z) + stddev*norm.pdf(z)
        score_multi.append(score_ind)
    score_multi = np.array(score_multi)
    if scoring=='sum':
        score = score_multi.sum(axis=0)
    # Todo, f1 score

    # ==========================================================================
    # Sort, if required
    if sort:
        sortix = np.argsort(score)[::-1]
        score, Candidates = score[sortix],Candidates[sortix]

    return score, Candidates

def EIGF_Multi(model, Candidates, scoring='sum',sort=True):
    NbOutput = len(model.models)
    # ==========================================================================
    # Use model to make prediction
    with torch.no_grad():
        _Candidates = torch.tensor(Candidates)
        output = model(*[_Candidates]*NbOutput)
    vars,diff = [],[]

    # ==========================================================================
    # Get nearest neighbour values (assumes same inputs for all dimensions)
    TrainIn = model.train_inputs[0][0].numpy()
    TrainOut = np.transpose([model.train_targets[i].numpy() for i in range(NbOutput)])
    NN = []
    for c in Candidates:
        d = np.linalg.norm(TrainIn - c,axis=1)
        NN.append(TrainOut[np.argmin(d)])
    NN = np.array(NN)

    # ==========================================================================
    # Calculate score & combine in to single value
    for i,out in enumerate(output):
        mean = out.mean.numpy()
        _diff = (mean - NN[:,i])**2
        diff.append(_diff)
        vars.append(out.variance.numpy())
    vars,diff = np.transpose(vars),np.transpose(diff)
    score_multi = vars+diff

    if scoring=='sum':
        score = score_multi.sum(axis=1)
    # Todo, f1 score

    # ==========================================================================
    # Sort, if required
    if sort:
        sortix = np.argsort(score)[::-1]
        score, Candidates = score[sortix],Candidates[sortix]

    return score, Candidates

def OptVar_Multi(Candidates,GPR_model,scoring='sum'):
    _Candidates = torch.tensor(Candidates)
    dvar, var = [], []
    for mod in GPR_model.models:
        _dvar, _var = mod.Gradient_variance(_Candidates)
        dvar.append(_dvar.detach().numpy())
        var.append(_var.detach().numpy())
    dvar, var = np.array(dvar),np.array(var)
    if scoring=='sum':
        var = var.sum(axis=0)
        dvar = dvar.sum(axis=0)

    return var, dvar

def _Constrain_Multi(Candidates,OrigPoint,rad):
    a = rad**2 - np.linalg.norm(Candidates - OrigPoint,axis=1)**2
    return a

def _dConstrain_Multi(Candidates,OrigPoint,rad):
    da = -2*(Candidates - OrigPoint)
    return da

def ConMax_Multi(model, Candidates, func, OrigPoint, rad=0.05, scoring='sum', sort=True):
    con1 = {'type': 'ineq', 'fun': _Constrain_Multi,
            'jac':_dConstrain_Multi, 'args':[OrigPoint,rad]}
    return Max_Multi(model,Candidates,func,scoring=scoring,sort=sort,constraints=(con1))

def Max_Multi(model, Candidates, func, scoring='sum',sort=True, constraints=()):
    NbOutput = len(model.models)
    NbInput = (model.train_inputs[0][0]).shape[1]
    order = 'decreasing' if sort else None
    if func.lower() == 'var': fn = OptVar_Multi

    Optima = FuncOpt(fn, Candidates, find='max', tol=None,
                     order=order, bounds=[[0,1]]*NbInput, jac=True,
                     constraints=constraints, args=[model])
    Candidates, score = Optima
    return score, Candidates

def EIGrad_Multi(model, Candidates, scoring='sum',sort=True):
    NbOutput = len(model.models)
    # ==========================================================================
    # Use model to make prediction

    _Candidates = torch.tensor(Candidates)
    dmean,var = [],[]
    for mod in model.models:
         _var = mod(_Candidates).variance
         _dmean, _mean = mod.Gradient_mean(_Candidates)
         dmean.append(_dmean.detach().numpy())
         var.append(_var.detach().numpy())
    dmean,var = np.array(dmean),np.array(var)


    # ==========================================================================
    # Get nearest neighbour values (assumes same inputs for all dimensions)
    TrainIn = model.train_inputs[0][0].numpy()
    NN = []
    for c in Candidates:
        diff = TrainIn - c
        d = np.linalg.norm(diff,axis=1)
        ix = np.argmin(d)
        NN.append(diff[ix])
    NN = np.array(NN)

    gradsc = (NN.T[:,:,None]*dmean.T)**2
    gradsc = gradsc.sum(axis=0).T
    score_multi = var + gradsc

    if scoring=='sum':
        score = score_multi.sum(axis=0)

    # ==========================================================================
    # Sort, if required
    if sort:
        sortix = np.argsort(score)[::-1]
        score, Candidates = score[sortix],Candidates[sortix]
    return score, Candidates
