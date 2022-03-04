import os
import sys
import numpy as np
import torch
import gpytorch
import h5py
from natsort import natsorted
import time

from .slsqp_multi import slsqp_multi

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
        Convergence,tol = [],1e-6
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
                if (i+1) % Print == 0:
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

            if i>2*ConvCheck and (i+1)%ConvCheck==0:
                mean_new = np.mean(Convergence[-ConvCheck:])
                mean_old = np.mean(Convergence[-2*ConvCheck:-ConvCheck])
                if mean_new>mean_old:
                    print('Terminating due to increasing loss')
                    break
                elif np.abs(mean_new-mean_old)<tol:
                    print('Terminating due to convergence')
                    break


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

def Create_GPR(TrainIn,TrainOut,Kernel,prev_state=None,min_noise=None):
    if TrainOut.ndim==1:
        # single output
        likelihood, model = _Create_GPR(TrainIn, TrainOut, Kernel, min_noise=min_noise)
    else:
        # multiple output
        NbModel = TrainOut.shape[1]
        # Change kernel and min_noise to a list
        if type(Kernel) not in (list,tuple): Kernel = [Kernel]*NbModel
        if type(min_noise) not in (list,tuple): min_noise = [min_noise]*NbModel
        models,likelihoods = [], []
        for i in range(NbModel):
            likelihood, model = _Create_GPR(TrainIn, TrainOut[:,i], Kernel[i],
                                            min_noise = min_noise[i])
            models.append(model)
            likelihoods.append(likelihood)

        model = gpytorch.models.IndependentModelList(*models)
        likelihood = gpytorch.likelihoods.LikelihoodList(*likelihoods)

    if prev_state:
        # Check that the file exists
        if os.path.isfile(prev_state):
            state_dict = torch.load(prev_state)
            model.load_state_dict(state_dict)
        else:
            print('Warning\nPrevious state file doesnt exist\nNo previous model is loaded\n')

    return likelihood, model

def _Create_GPR(TrainIn,TrainOut,Kernel,prev_state=None,min_noise=None):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPmodel(TrainIn, TrainOut, likelihood, Kernel)
    if not prev_state:
        if min_noise != None:
            likelihood.noise_covar.register_constraint('raw_noise',gpytorch.constraints.GreaterThan(min_noise))

        # Start noise at lower level to avoid bad optima, but ensure it isn't zero
        noise_lower = likelihood.noise_covar.raw_noise_constraint.lower_bound
        noise_init = max(5*noise_lower,1e-8)
        hypers = {'likelihood.noise_covar.noise': noise_init}
        model.initialize(**hypers)
    else:
        state_dict = torch.load(prev_state)
        model.load_state_dict(state_dict)

    return likelihood, model

# ==============================================================================
# Train the GPR model
def GPR_Train(model, Epochs=5000, lr=0.01, Print=50, ConvCheck=50, Verbose=False, TestData=None,):
    likelihood = model.likelihood

    model.train()
    likelihood.train()

    MultiOutput = True if hasattr(model,'models') else False
    SumMLL = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    if MultiOutput and not SumMLL:
        _mll = gpytorch.mlls.ExactMarginalLogLikelihood
        mll,Convergence = [],[]
        for mod in model.models:
            mll.append(_mll(mod.likelihood,mod))
            Convergence.append([])
    else:
        Convergence = []
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        if MultiOutput:
            mll = gpytorch.mlls.SumMarginalLogLikelihood(likelihood, model)
        else:
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood,model)

    tol = 1e-6
    for i in range(Epochs):
        optimizer.zero_grad()

        if MultiOutput and not SumMLL:
            TotalLoss = 0
            for j,mod in enumerate(model.models):
                output = mod(*model.train_inputs[j])

                _loss = -mll[j](output,model.train_targets[j])
                TotalLoss+=_loss.item()
                Convergence[j].append(_loss.item())

                _loss.backward()

            TotalLoss /=(j+1)
        else:
            output = model(*model.train_inputs)
            loss = -mll(output, model.train_targets)
            TotalLoss = loss.item()
            Convergence.append(TotalLoss)

            loss.backward()

        optimizer.step()

        if i>2*ConvCheck and (i+1)%ConvCheck==0:
            mean_new = np.mean(Convergence[-ConvCheck:])
            mean_old = np.mean(Convergence[-2*ConvCheck:-ConvCheck])
            if mean_new>mean_old:
                print('Terminating due to increasing loss')
                # break
            elif np.abs(mean_new-mean_old)<tol:
                print('Terminating due to convergence')
                # break

        if i==0 or (i+1) % Print == 0:
            message = "Iteration: {}, Loss: {}".format(i+1,TotalLoss)
            # if Verbose:
            #     for i,mod in enumerate(model.models):
            #         Lscales = mod.covar_module.base_kernel.lengthscale.detach().numpy()[0]
            #         Outscale = mod.covar_module.outputscale.detach().numpy()
            #         noise = mod.likelihood.noise.detach().numpy()[0]
            #         message += "Model output {}:\nLength scale: {}\nOutput scale: {}\n"\
            #                   "Noise: {}\n".format(i,Lscales,Outscale,noise)
            print(message)

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

    print("\nFinal model parameters")
    Modstr = "Length scales: {}\nOutput scale: {}\n Noise: {}\n"
    if MultiOutput:
        for i,mod in enumerate(model.models):
            LS = mod.covar_module.base_kernel.lengthscale.detach().numpy()[0]
            OS = mod.covar_module.outputscale.detach().numpy()
            N = mod.likelihood.noise.detach().numpy()[0]
            print(("Output {}\n"+Modstr).format(i,LS,OS,N))
    else:
            LS = model.covar_module.base_kernel.lengthscale.detach().numpy()[0]
            OS = model.covar_module.outputscale.detach().numpy()
            N = model.likelihood.noise.detach().numpy()[0]
            print(Modstr.format(LS,OS,N))

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

def Openhdf(File,style,timer=5):
    ''' Repeatedly attemps to open hdf file if it is held by another process for
    the time allocated by timer '''
    st = time.time()
    while True:
        try:
            Database = h5py.File(File,style)
            return Database
        except OSError:
            if time.time() - st > timer:
                sys.exit('Timeout on opening hdf file')

def Writehdf(File, array, data_path):
    Database = Openhdf(File,'a')
    if data_path in Database:
        del Database[data_path]
    Database.create_dataset(data_path,data=array)
    Database.close()

def Readhdf(File, data_paths):
    Database = Openhdf(File,'r')

    if type(data_paths)==str: data_paths = [data_paths]
    data = []
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
# ML model Optima

def GetOptima(model, NbInit, bounds, seed=None, find='max', tol=0.01,
              order='decreasing', success_only=True, constraints=()):
    if seed!=None: np.random.seed(seed)
    init_points = np.random.uniform(0,1,size=(NbInit,len(bounds)))

    Optima = slsqp_multi(_GPR_Opt, init_points, bounds=bounds,
                         constraints=constraints,find=find, tol=tol,
                         order=order, success_only=success_only,
                         jac=True, args=[model])
    Optima_cd, Optima_val = Optima
    return Optima_cd, Optima_val

def GetExtrema(model,NbInit,bounds,seed=None):
    # ==========================================================================
    # Get min and max values for each
    Extrema_cd, Extrema_val = [], []
    for tp,order in zip(['min','max'],['increasing','decreasing']):
        _Extrema_cd, _Extrema_val = GetOptima(model, NbInit, bounds,seed,
                                              find=tp, order=order)
        Extrema_cd.append(_Extrema_cd[0])
        Extrema_val.append(_Extrema_val[0])
    return np.array(Extrema_val), np.array(Extrema_cd)

def _GPR_Opt(X,model):
    torch.set_default_dtype(torch.float64)
    X = torch.tensor(X)
    dmean, mean = model.Gradient_mean(X)
    return mean.detach().numpy(), dmean.detach().numpy()
