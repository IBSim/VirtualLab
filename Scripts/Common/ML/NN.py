import os

import numpy as np
import torch

from . import ML
from Scripts.Common import VLFunctions as VLF

def BuildModel(TrainData, TestData, ModelDir, ModelParameters={},
             TrainingParameters={}, FeatureNames=None,LabelNames=None):

    TrainIn,TrainOut = TrainData
    # Create dataspace to conveniently keep data together
    Dataspace = ML.DataspaceTrain(TrainData,Test=TestData)

    # ==========================================================================
    # get model
    # Add input and output to architecture

    if 'Architecture' not in ModelParameters:
        sys.exit('Must have architecture')

    _architecture = ModelParameters.pop('Architecture')
    _architecture = AddIO(_architecture,Dataspace)

    model = NN_FC(Architecture=_architecture, **ModelParameters)

    NbWeights = GetNbWeights(model)
    ML.ModelSummary(Dataspace.NbInput,Dataspace.NbOutput,Dataspace.NbTrain,
                    NbWeights=NbWeights, Features=FeatureNames,
                    Labels=LabelNames)

    # Train model
    Convergence = TrainModel(model, Dataspace, **TrainingParameters)
    model.eval()

    SaveModel(ModelDir,model,TrainIn,TrainOut,Convergence)

    return  model, Dataspace

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

    np.save("{}/Convergence".format(ModelDir),Convergence)

def LoadModel(ModelDir):
    TrainIn = np.load("{}/Input.npy".format(ModelDir))
    TrainOut = np.load("{}/Output.npy".format(ModelDir))
    Dataspace = ML.DataspaceTrain([TrainIn,TrainOut])
    if os.path.isfile("{}/Parameters.py".format(ModelDir)):
        Parameters = VLF.ReadParameters("{}/Parameters.py".format(ModelDir))
        ModelParameters = getattr(Parameters,'ModelParameters',{})
    else:
        ModelParameters, Parameters = {},None


    _architecture = ModelParameters.pop('Architecture')
    _architecture = AddIO(_architecture,Dataspace)

    model = NN_FC(Architecture=_architecture,**ModelParameters)

    state_dict = torch.load("{}/Model.pth".format(ModelDir))
    model.load_state_dict(state_dict)

    model.eval()

    return model, Dataspace, Parameters

# def _loss_func_weight(weights):
#     weights = torch.from_numpy(weights)
#     def _loss_wrap(output,target):
#         loss_all = torch.nn.MSELoss(reduction='none')(output,target)
#         loss = (loss_all*weights).sum()
#         return loss
#     return _loss_wrap

def TrainModel(model, Dataspace, Epochs=5000, lr=0.005, batch_size=32, Print=50,
               ConvStart=None, ConvAvg=10, tol=1e-4, norm_outputs=True,
               Verbose=False):

    if norm_outputs:
        TrainOut,TestOut = Dataspace.TrainOut_scale,Dataspace.TestOut_scale
    else:
        TrainOut = ML.DataRescale(Dataspace.TrainOut_scale.detach().numpy(),*Dataspace.OutputScaler)
        Dataspace.TrainOut_scale = torch.from_numpy(TrainOut)
        TestOut = ML.DataRescale(Dataspace.TestOut_scale.detach().numpy(),*Dataspace.OutputScaler)
        Dataspace.TestOut_scale = torch.from_numpy(TestOut)

    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_func = torch.nn.MSELoss(reduction='mean')

    train_dataset = torch.utils.data.TensorDataset(Dataspace.TrainIn_scale, Dataspace.TrainOut_scale)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    loss_train_list, loss_test_list = [],[]
    for epoch in range(0,Epochs):
        # Loop through the batches for each epoch
        for step, (batch_x, batch_y) in enumerate(train_loader):
            # forward and loss
            loss = loss_func(model(batch_x), batch_y)
            # backwardTestOut.dot(VT.T)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validate
        model.eval() # Change to eval to switch off gradients and dropout
        with torch.no_grad():
            loss_train = loss_func(model(Dataspace.TrainIn_scale), Dataspace.TrainOut_scale).detach().numpy()
            loss_test = loss_func(model(Dataspace.TestIn_scale), Dataspace.TestOut_scale).detach().numpy()
            loss_test_list.append(loss_test); loss_train_list.append(loss_train)
        model.train()

        if epoch==0 or (epoch+1)%Print==0:
            print("Epoch:{}, Train loss:{:.4e}, Test loss:{:.4e}".format(epoch,loss_train,loss_test))

    return loss_train_list, loss_test_list


def Performance(model,Data, mean=False):
    for name,(data_in,data_out) in Data.items():
        with torch.no_grad():
            pred = model(data_in).numpy()
        metric = ML.GetMetrics2(pred,data_out.detach().numpy())

        print('==============================================================')
        if mean:
            print('{}\n{}\n'.format(name,metric.mean()))
        else:
            print('{}\n{}\n'.format(name,metric))

def Performance_PCA(model,Data, VT, OutputScaler, mean=True):
    ''' Gets the averages of the metrics for the fully scaled version '''

    for name,(data_in,data_out) in Data.items():
        with torch.no_grad():
            pred = model(data_in).numpy()

        pred_rescale = ML.DataRescale(pred,*OutputScaler)
        data_out_rescale = ML.DataRescale(data_out.detach().numpy(),*OutputScaler)

        metric = ML.GetMetrics2(pred_rescale.dot(VT),data_out_rescale.dot(VT))

        print('==============================================================')
        if mean:
            print('{}\n{}\n'.format(name,metric.mean()))
        else:
            print('{}\n{}\n'.format(name,metric))

def GetNbWeights(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def AddIO(Architecture,Dataspace):
    return [Dataspace.NbInput] + Architecture + [Dataspace.NbOutput]

# ==============================================================================
# Model wrappers

def GetModel(model_dir):
    model, Dataspace, Parameters = LoadModel(model_dir)
    mod_wrap = ModelWrap(model,Dataspace)
    return mod_wrap

def GetModelPCA(model_dir):
    model, Dataspace, Parameters = LoadModel(model_dir)
    VT = np.load("{}/VT.npy".format(model_dir))
    ScalePCA = np.load("{}/ScalePCA.npy".format(model_dir))
    mod_wrap = ModelWrapPCA(model,Dataspace,VT,ScalePCA)
    return mod_wrap

class ModelWrap(ML.ModelWrapBase):
    def Predict(self,inputs, scale_outputs=True):
        if type(inputs) is not torch.Tensor:
            inputs = torch.from_numpy(inputs)

        with torch.no_grad():
            pred = self.model(inputs).numpy()

        if scale_outputs:
            pred = ML.DataRescale(pred,*self.Dataspace.OutputScaler)

        return pred

class ModelWrapPCA(ModelWrap):
    def __init__(self,model,Dataspace,VT,ScalePCA):
        super().__init__(model,Dataspace)
        self.VT = VT
        self.ScalePCA = ScalePCA

    def PredictFull(self,inputs,scale_outputs=True):
        PC_pred = self.Predict(inputs,scale_outputs=True) # get prediction on PCs
        FullPred = self.Reconstruct(PC_pred,scale=scale_outputs)
        return FullPred

    def PredictFull_dset(self,dset_name,scale_outputs=True):
        inputs = self.GetDatasetInput(dset_name)
        return self.PredictFull(inputs,scale_outputs=scale_outputs)

    def Compress(self,output,scale=True):
        if scale:
            output = ML.DataScale(output,*self.ScalePCA)
        return output.dot(self.VT.T)

    def Reconstruct(self,PC,scale=True):
        recon = PC.dot(self.VT)
        if scale:
            recon = ML.DataRescale(recon,*self.ScalePCA)
        return recon

# ==============================================================================
# Vanilla NN
class NN_FC(torch.nn.Module):
    def __init__(self, Architecture, Dropout=None):
        super(NN_FC, self).__init__()

        self.Architecture = Architecture
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
                x = torch.nn.functional.leaky_relu(x)
        return x


    def OptimiseOutput(self,x):
        x.requires_grad=True
        pred = self(x)[:,self.output_ix]
        grads = torch.autograd.grad(pred.sum(), x)[0]
        return pred, grads

    def _Gradient(self, input):
        '''
        Function which returns the NN gradient at N input points
        Input: 2-d array of points with dimension (N ,NbInput)
        Output: 3-d array of partial derivatives (NbOutput, NbInput, N)
        '''
        input = np.atleast_2d(input)
        for i in range(self.NbCnct):
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
            if i==0:
                Cumul = layergrad
            else :
                Cumul = np.einsum('ikj,ilk->ilj', Cumul, layergrad)

        return Cumul
