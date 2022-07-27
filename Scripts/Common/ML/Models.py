import os

import numpy as np
import torch
import torch.nn.functional as F
import gpytorch

# ==============================================================================
# Exact GPR model
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

# ==============================================================================
# Multitask GPR model
class MultitaskGPModel(gpytorch.models.ExactGP):
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
# Vanilla NN
class NN_FC(torch.nn.Module):
    def __init__(self, Architecture, Dropout=None):
        super(NN_FC, self).__init__()

        self.Dropout = Dropout
        self.NbCnct = len(Architecture) - 1

        for i in range(self.NbCnct):
            fc = torch.nn.Linear(Architecture[i],Architecture[i+1])
            setattr(self,"fc{}".format(i),fc)

    def forward(self, x):
        # for i, drop in enumerate(self.Dropout[:-1]):
        for i in range(self.NbCnct):
            # x = nn.Dropout(drop)(x)
            fc = getattr(self,"fc{}".format(i))
            x = fc(x)
            if i < self.NbCnct-1:
                x = F.leaky_relu(x)
        return x

    def Gradient(self, input):
        '''
        Function which returns the NN gradient at N input points
        Input: 2-d array of points with dimension (N ,NbInput)
        Output: 3-d array of partial derivatives (NbOutput, NbInput, N)
        '''
        input = np.atleast_2d(input)
        for i in range(1,self.NbLayers):
            fc = getattr(self,'fc{}'.format(i))
            w = fc.weight.detach().numpy()
            b = fc.bias.detach().numpy()
            out = np.einsum('ij,kj->ik',input, w) + b

            # create relu function
            out[out<0] = out[out<0]*0.01
            input = out
            # create relu gradients
            diag = np.copy(out)
            diag[diag>=0] = 1
            diag[diag<0] = 0.01

            layergrad = np.einsum('ij,jk->ijk',diag,w)
            if i==1:
                Cumul = layergrad
            else :
                Cumul = np.einsum('ikj,ilk->ilj', Cumul, layergrad)

        return Cumul
