import os

import numpy as np
import torch
import torch.nn.functional as F
import gpytorch

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
