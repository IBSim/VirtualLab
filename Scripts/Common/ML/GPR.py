import os
import sys
import time

import numpy as np
import torch
import gpytorch

from Scripts.Common import VLFunctions as VLF
from . import ML

# ==============================================================================
# Generic building function for GPR
def BuildModel(TrainData, TestData, ModelDir, ModelParameters={},
             TrainingParameters={}, FeatureNames=None,LabelNames=None):

    TrainIn,TrainOut = TrainData
    TestIn,TestOut = TestData

    Dataspace = ML.DataspaceTrain(TrainData,Test=TestData)

    # ==========================================================================
    # Model summary

    ML.ModelSummary(Dataspace.NbInput,Dataspace.NbOutput,Dataspace.NbTrain,
                    TestNb=TestIn.shape[0], Features=FeatureNames,
                    Labels=LabelNames)

    # ==========================================================================
    # get model & likelihoods
    likelihood, model = Create_GPR(Dataspace.TrainIn_scale, Dataspace.TrainOut_scale,
                                   **ModelParameters,
                                   input_scale=Dataspace.InputScaler,
                                   output_scale=Dataspace.OutputScaler)

    # Train model
    Convergence = TrainModel(model, **TrainingParameters)
    model.eval()

    SaveModel(ModelDir,model,TrainIn,TrainOut,Convergence)

    return  likelihood, model, Dataspace

# ==============================================================================
# Functions for saving & loading models
def SaveModel(ModelDir,model,TrainIn,TrainOut,Convergence):
    ''' Function to store model infromation'''
    # ==========================================================================
    # Save information
    os.makedirs(ModelDir,exist_ok=True)
    # save data
    np.save("{}/Input".format(ModelDir),TrainIn)
    np.save("{}/Output".format(ModelDir), TrainOut)

    # save model
    ModelFile = "{}/Model.pth".format(ModelDir)
    torch.save(model.state_dict(), ModelFile)

    # Plot convergence & save
    conv_len = [len(c) for c in Convergence]
    conv_sum = np.zeros(max(conv_len))
    for c in Convergence:
        conv_sum[:len(c)]+=np.array(c)
        conv_sum[len(c):]+=c[-1]
    np.save("{}/Convergence".format(ModelDir),conv_sum)

def LoadModel(ModelDir):
    ''' Function which loads GPR model from ModelDir.'''
    TrainIn = np.load("{}/Input.npy".format(ModelDir))
    TrainOut = np.load("{}/Output.npy".format(ModelDir))
    Dataspace = ML.DataspaceTrain([TrainIn,TrainOut])

    Parameters = VLF.ReadParameters("{}/Parameters.py".format(ModelDir))
    ModelParameters = getattr(Parameters,'ModelParameters',{})
    likelihood, model = Create_GPR(Dataspace.TrainIn_scale, Dataspace.TrainOut_scale,
                                   input_scale=Dataspace.InputScaler,
                                   output_scale=Dataspace.OutputScaler,
                                   prev_state="{}/Model.pth".format(ModelDir),
                                   **ModelParameters)
    model.eval()

    return  likelihood, model, Dataspace, Parameters
# ==============================================================================
# Functions to easily create GPR
def Create_GPR(TrainIn,TrainOut,kernel='RBF',prev_state=None,min_noise=None,input_scale=None,
                output_scale=None,multitask=False):
    if TrainOut.ndim==1:
        # single output
        likelihood, model = _SingleGPR(TrainIn, TrainOut, kernel, min_noise=min_noise,
                                        input_scale=input_scale,output_scale=output_scale)

    elif multitask:
        likelihood, model = _MultitaskGPR(TrainIn,TrainOut,kernel,min_noise=min_noise,
                                        input_scale=input_scale,output_scale=output_scale)
    else:
        # multiple output
        NbModel = TrainOut.shape[1]
        # Change kernel and min_noise to a list
        if type(kernel) not in (list,tuple): kernel = [kernel]*NbModel
        if type(min_noise) not in (list,tuple): min_noise = [min_noise]*NbModel
        models,likelihoods = [], []
        for i in range(NbModel):
            _output_scale = output_scale[:,i] if output_scale is not None else None
            likelihood, model = _SingleGPR(TrainIn, TrainOut[:,i], kernel[i], min_noise = min_noise[i],
                                            input_scale=input_scale, output_scale=_output_scale)
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

def _SingleGPR(TrainIn,TrainOut,kernel,prev_state=None,min_noise=None,input_scale=None,output_scale=None):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPmodel(TrainIn, TrainOut, likelihood, kernel)
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

    if input_scale is not None: model.input_scale = input_scale
    if output_scale is not None: model.output_scale = output_scale

    return likelihood, model


def _MultitaskGPR(TrainIn,TrainOut,kernel,prev_state=None,min_noise=None,input_scale=None,output_scale=None):
    ndim = TrainOut.shape[1]
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=ndim)
    model = MultitaskGPModel(TrainIn, TrainOut, likelihood, kernel,rank=1)

    if input_scale is not None: model.input_scale = input_scale
    if output_scale is not None: model.output_scale = output_scale

    return likelihood, model

# ==============================================================================
# Train the GPR model
def TrainModel(model, Epochs=5000, lr=0.01, Print=50, ConvAvg=10, tol=1e-4,
              Verbose=False, SumOutput=False):

    likelihood = model.likelihood
    model.train()
    likelihood.train()

    MultiOutput = True if hasattr(model,'models') else False
    # Create the necessary loss functions and optimisers
    if MultiOutput and not SumOutput:
        # Each output of the model is trained seperately
        TrackLoss, Completed = 0, []
        _mll = gpytorch.mlls.ExactMarginalLogLikelihood
        LossFn,Losses, optimizer = [],[],[]
        for mod in model.models:
            LossFn.append(_mll(mod.likelihood,mod))
            optimizer.append(torch.optim.Adam(mod.parameters(), lr=lr))
            Losses.append([])
    elif MultiOutput:
        # Each output is trained together
        Losses = [[]]
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        LossFn = gpytorch.mlls.SumMarginalLogLikelihood(likelihood, model)
    else:
        # Single output model
        Losses = [[]]
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        LossFn = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood,model)

    # Start looping over training data
    for i in range(Epochs):
        if MultiOutput and not SumOutput:
            TotalLoss = TrackLoss
            for j,mod in enumerate(model.models):
                if j in Completed: continue

                _Losses = Losses[j]
                _convergence = _Step(mod,optimizer[j],LossFn[j],_Losses,ConvAvg=ConvAvg,tol=tol)
                TotalLoss+=_Losses[-1]

                if _convergence != None:
                    Completed.append(j)
                    TrackLoss+=_Losses[-1]
                    print("Output {}: {}".format(j,_convergence))

            TotalLoss = TotalLoss/(j+1)
            if len(Completed)==j+1:
                break
        else:
            _Losses = Losses[0]
            convergence = _Step(model,optimizer,LossFn,_Losses,ConvAvg=ConvAvg,tol=tol)
            TotalLoss = _Losses[-1]
            if convergence != None:
                print(convergence)
                break

        if i==0 or (i+1) % Print == 0:
            print("Iteration: {}, Loss: {}".format(i+1,TotalLoss))
            if Verbose:
                PrintParameters(model)

    # Print out final information about training & model parameters
    # print('\n################################\n')
    # print("Iterations: {}\nLoss: {}".format(i+1,TotalLoss))
    # print("\nModel parameters:")
    # PrintParameters(model)
    # print('################################\n')

    return Losses

def _Step(model, optimizer, mll, loss_lst, ConvAvg=10, tol=1e-4):
    optimizer.zero_grad() # set all gradients to zero
    # Calculate loss & add to list

    output = model(*model.train_inputs)

    loss = -mll(output, model.train_targets)
    loss_lst.append(loss.item())
    # Check convergence using the loss list. If convergence, return
    convergence = CheckConvergence(loss_lst,ConvAvg=ConvAvg,tol=tol)
    if convergence != None:
        return convergence

    # Calculate gradients & update model parameters using the optimizer
    loss.backward()
    optimizer.step()

def CheckConvergence(Loss, ConvAvg=10, tol=1e-4):
    ''' Checks the list of loss values to decide whether or not convergence has been reached'''
    if len(Loss) >= 2*ConvAvg:
        mean_new = np.mean(Loss[-ConvAvg:])
        mean_old = np.mean(Loss[-2*ConvAvg:-ConvAvg])
        if mean_new > mean_old:
            return "Convergence reached after {} iterations. Loss increasing".format(len(Loss))
        elif np.abs(mean_new-mean_old)<tol:
            return "Convergence reached after {} iterations. Loss change smaller than tolerance".format(len(Loss))

def PrintParameters(model):
    Modstr = "Length scales: {}\nOutput scale: {}\nNoise: {}\n\n"
    if hasattr(model,'models'):
        Rstr = ""
        for i,mod in enumerate(model.models):
            LS = mod.covar_module.base_kernel.lengthscale.detach().numpy()[0]
            OS = mod.covar_module.outputscale.detach().numpy()
            N = mod.likelihood.noise.detach().numpy()[0]
            Rstr += ("Output {}\n"+Modstr).format(i,LS,OS,N)
    else:
            LS = model.covar_module.base_kernel.lengthscale.detach().numpy()[0]
            OS = model.covar_module.outputscale.detach().numpy()
            N = model.likelihood.noise.detach().numpy()[0]
            Rstr = Modstr.format(LS,OS,N)

    print(Rstr,end='')


# ==============================================================================
# GPR models

class ExactGPmodel(gpytorch.models.ExactGP):
    ''' Exact GPR model. '''
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

class MultitaskGPModel(gpytorch.models.ExactGP):
    ''' Multitask GPR model. '''
    def __init__(self, train_x, train_y, likelihood,kernel,options={},ard=True,rank=1):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)

        ard_num_dims = train_x.shape[1] if ard else None
        ndim = train_y.shape[1]

        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=ndim
        )

        if kernel.lower() in ('rbf'):
            _kernel = gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims)
        if kernel.lower().startswith('matern'):
            split = kernel.split('_')
            nu = float(split[1]) if len(split)==2 else 2.5
            _kernel = gpytorch.kernels.MaternKernel(nu=nu,ard_num_dims=ard_num_dims)

        self.covar_module = gpytorch.kernels.MultitaskKernel(
            _kernel, num_tasks=ndim, rank=rank
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

# ==============================================================================
