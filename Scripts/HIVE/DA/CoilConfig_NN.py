import os
import sys
import h5py
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib import cm

from types import SimpleNamespace as Namespace
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data

from Scripts.Common.VLFunctions import MeshInfo
from Functions import Uniformity2 as UniformityScore, DataScale, DataRescale
from Optimise import FuncOpt
from Sim.PreHIVE import ERMES

def Single(VL, DADict):
    ML = DADict["Parameters"]

    NbTorchThread = getattr(ML,'NbTorchThread',1)
    torch.set_num_threads(NbTorchThread)
    torch.manual_seed(getattr(ML,'Seed',100))

    # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')

    # File where all data is stored
    DataFile = "{}/ML/Data.hdf5".format(VL.PROJECT_DIR)

    if ML.Train:
        DataSplit = getattr(ML,'DataSplit',0.7)
        TrainNb = getattr(ML,'TrainNb',None)

        #=======================================================================
        # Get Training & Testing data from file
        MLData = h5py.File(DataFile,'r')

        if ML.TrainData in MLData:
            _TrainData = MLData[ML.TrainData][:]
        elif "{}/{}".format(VL.StudyName,ML.TrainData) in MLData:
            _TrainData = MLData["{}/{}".format(VL.StudyName,ML.TrainData)][:]
        else : sys.exit("Training data not found")

        if not TrainNb:
            TrainNb = int(np.ceil(_TrainData.shape[0]*DataSplit))
        TrainData = _TrainData[:TrainNb,:] # Data used for training

        np.save("{}/TrainData".format(DADict["CALC_DIR"]),TrainData)

        TestNb = int(np.ceil(TrainNb*(1-DataSplit)/DataSplit))
        if hasattr(ML,'TestData'):
            if ML.TestData == ML.TrainData:
                TestData = _TrainData[TrainNb:,:]
            elif ML.TestData in MLData :
                TestData = MLData[ML.TestData][:]
            else : sys.exit('Testing data not found')
        else:
            TestData = _TrainData[TrainNb:,:]
        TestData = TestData[:TestNb,:]

        np.save("{}/TestData".format(DADict["CALC_DIR"]),TestData)

        MLData.close()

    else:
        TrainData = np.load("{}/TrainData.npy".format(DADict["CALC_DIR"]))
        TestData = np.load("{}/TestData.npy".format(DADict["CALC_DIR"]))

    #=======================================================================

    # Convert data to float32 (needed for pytorch)
    TrainData = TrainData.astype('float32')
    Train_x, Train_y = TrainData[:,:4], TrainData[:,4:]

    # Scale test & train input data to [0,1] (based on training data)
    InputScaler = np.array([Train_x.min(axis=0),Train_x.max(axis=0) - Train_x.min(axis=0)])
    # InputScaler = np.array([np.mean(Train_x,axis=0),np.std(Train_x,axis=0)])
    Train_x_scale = DataScale(Train_x,*InputScaler)
    # Scale test & train output data to [0,1] (based on training data)
    OutputScaler = np.array([Train_y.min(axis=0),Train_y.max(axis=0) - Train_y.min(axis=0)])
    # OutputScaler = np.array([np.mean(Train_y,axis=0),np.std(Train_y,axis=0)])
    Train_y_scale = DataScale(Train_y,*OutputScaler)

    Train_x_tf = torch.from_numpy(Train_x_scale)
    Train_y_tf = torch.from_numpy(Train_y_scale)

    TestData = TestData.astype('float32')
    Test_x, Test_y = TestData[:,:4], TestData[:,4:]
    Test_x_scale = DataScale(Test_x,*InputScaler)
    Test_y_scale = DataScale(Test_y,*OutputScaler)
    Test_x_tf = torch.from_numpy(Test_x_scale)
    Test_y_tf = torch.from_numpy(Test_y_scale)

    ModelFile = '{}/model.h5'.format(DADict["CALC_DIR"]) # File model will
    model = NetPU(ML.NNLayout,ML.Dropout)


    if ML.Train:
        GPU = getattr(ML,'GPU', False)
        lr = getattr(ML,'lr', 0.0001)
        PrintEpoch = getattr(ML,'PrintEpoch',100)
        ConvCheck = getattr(ML,'ConvCheck',100)
        # DADict['Data']['MSE'] = MSEvals = {}

        model.train()

        # Create batches of the data
        train_dataset = Data.TensorDataset(Train_x_tf, Train_y_tf)
        train_loader = Data.DataLoader(train_dataset, batch_size=ML.BatchSize, shuffle=True)

        # optimizer
        optimizer = optim.AdamW(model.parameters(), lr=lr)

        # loss tensor

        loss_func = nn.MSELoss(reduction='mean')
        loss_func_split = nn.MSELoss(reduction='none')
        # Convergence history
        LossConv = {'loss_train': [], 'loss_test': []}
        LossConvSplit = {'loss_train': [], 'loss_test': []}
        BestModel = copy.deepcopy(model)
        BestLoss_test, OldAvg = float('inf'), float('inf')
        print("Starting training")
        print("Training set: {}\nTest set: {}\n".format(Train_x_tf.size()[0], Test_x_tf.size()[0]))
        print("{:8}{:12}{:12}".format('Epoch','Train_loss','Val_loss'))
        for epoch in np.arange(1,ML.NbEpoch+1):
            # Loop through the batches for each epoch
            for step, (batch_x, batch_y) in enumerate(train_loader):
                # forward and loss
                loss = loss_func(model(batch_x), batch_y)
                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # validate
            model.eval() # Change to eval to switch off gradients and dropout
            with torch.no_grad():
                loss_test = loss_func(model(batch_x), batch_y)
                loss_train = loss_func(model(Train_x_tf), Train_y_tf)

                loss_test_split = loss_func_split(model(Test_x_tf), Test_y_tf).numpy().mean(axis=0)
                loss_train_split = loss_func_split(model(Train_x_tf), Train_y_tf).numpy().mean(axis=0)

            model.train()

            LossConv['loss_test'].append(loss_test.cpu().detach().numpy().tolist()) # loss of full test
            LossConv['loss_train'].append(loss_train.cpu().detach().numpy().tolist()) #loss of full train

            LossConvSplit['loss_test'].append(loss_test_split.tolist()) # loss of full test
            LossConvSplit['loss_train'].append(loss_train_split.tolist()) #loss of full train

            if (epoch) % PrintEpoch == 0:
                print("{:<8}{:<12}{:<12}".format(epoch,"%.8f" % loss_train,"%.8f" % loss_test))

            if loss_test < BestLoss_test:
                BestLoss_train = loss_train
                BestLoss_test = loss_test
                BestModel = copy.deepcopy(model)

            if (epoch) % ConvCheck == 0:
                Avg = np.mean(LossConv['loss_test'][-ConvCheck:])
                if Avg > OldAvg:
                    print("Training terminated due to convergence")
                    # break
                OldAvg = Avg

        model = BestModel

        print('Training complete\n')

        SepLoss = nn.MSELoss(reduction='none')
        with torch.no_grad():
            Train = torch.mean(SepLoss(model(Train_x_tf), Train_y_tf),dim=0).numpy()
            Test = torch.mean(SepLoss(model(Test_x_tf), Test_y_tf),dim=0).numpy()
        Train_MSE_P,Train_MSE_V = Train*OutputScaler[1,:]**2
        Test_MSE_P,Test_MSE_V = Test*OutputScaler[1,:]**2
        print(Train_MSE_P,Test_MSE_P)

        print("Training loss: {:.8f}\nValidation loss: {:.8f}".format(BestLoss_train,BestLoss_test))

        # save model & training/testing data
        torch.save(model.state_dict(), ModelFile)
        fnt = 36

        plt.figure(figsize=(15,10))
        plt.plot(LossConv['loss_train'][0:epoch], label='Train', c=plt.cm.gray(0))
        plt.plot(LossConv['loss_test'][0:epoch], label='Test', c=plt.cm.gray(0.5))
        plt.ylabel("Loss (MSE)",fontsize=fnt)
        plt.xlabel("Epochs",fontsize=fnt)
        plt.legend(fontsize=fnt)
        plt.savefig("{}/Convergence.eps".format(DADict["CALC_DIR"]),dpi=600)
        plt.close()

        np.save("{}/Train_C".format(DADict['CALC_DIR']),LossConv['loss_train'][0:epoch])
        np.save("{}/Test_C".format(DADict['CALC_DIR']),LossConv['loss_test'][0:epoch])

        if True:
            plt.figure(figsize=(15,10))
            LossTrainSplit = np.array(LossConvSplit['loss_train'])
            LossTestSplit = np.array(LossConvSplit['loss_test'])
            plt.plot(LossTrainSplit[0:epoch,0], label='Train Power',c=plt.cm.gray(0))
            plt.plot(LossTestSplit[0:epoch,0], label='Test Power',c=plt.cm.gray(0.2))
            plt.plot(LossTrainSplit[0:epoch,1], label='Train Variation',c=plt.cm.gray(0.4))
            plt.plot(LossTestSplit[0:epoch,1], label='Test Variation',c=plt.cm.gray(0.6))
            plt.ylabel("Loss (MSE)",fontsize=fnt)
            plt.xlabel("Epochs",fontsize=fnt)
            plt.legend(fontsize=fnt)
            plt.savefig("{}/Convergence_Split.eps".format(DADict["CALC_DIR"]),dpi=600)
            plt.close()

            np.save("{}/Train_P".format(DADict['CALC_DIR']),LossTrainSplit[0:epoch,0])
            np.save("{}/Test_P".format(DADict['CALC_DIR']),LossTestSplit[0:epoch,0])
            np.save("{}/Train_V".format(DADict['CALC_DIR']),LossTrainSplit[0:epoch,1])
            np.save("{}/Test_V".format(DADict['CALC_DIR']),LossTestSplit[0:epoch,1])
        # # Predicted values from test and train data
        # with torch.no_grad():
        #     NN_train = model.predict(In_train)
        #     NN_test = model.predict(In_test)
        # NN_train = DataDenorm(NN_train, OutputRange)
        # NN_test = DataDenorm(NN_test, OutputRange)

    else:
        model.load_state_dict(torch.load(ModelFile))

    model.eval()

    if getattr(ML,'Input',None) != None:
        with torch.no_grad():
            x_scale = DataScale(np.atleast_2d(ML.Input),*InputScaler)
            x_scale = torch.tensor(x_scale, dtype=torch.float32)
            y = Power(x_scale)
            y = DataRescale(y.numpy(),*OutputScaler[:,0])
            print(y)
            DADict['Output'] = y.tolist()

    # Get bounds of data for optimisation
    bnds = list(zip(Train_x_scale.min(axis=0), Train_x_scale.max(axis=0)))

    MeshFile = "{}/AMAZEsample.med".format(VL.MESH_DIR)
    ERMES_Parameters = {'CoilType':'HIVE',
                        'Current':1000,
                        'Frequency':1e4,
                        'Materials':{'Block':'Copper_NL', 'Pipe':'Copper_NL', 'Tile':'Tungsten_NL'}}

    # Optimsation 1: Find the point of max power
    # Find the point(s) which give the maximum power
    print("Locating optimum configuration(s) for maximum power")
    NbInit = ML.MaxPowerOpt.get('NbInit',20)
    np.random.seed(123)
    init_guess = np.random.uniform(0,1,size=(NbInit,4))

    Optima = FuncOpt(NN_Opt, init_guess, find='max', tol=0.01, order='decreasing',
                     bounds = bnds, jac=True, args=[model,0])
    MaxPower_cd,MaxPower_val = Optima

    with torch.no_grad():
        MaxPower_cd_tf = torch.tensor(MaxPower_cd, dtype=torch.float32)
        NN_Out = model(MaxPower_cd_tf).numpy()

    MaxPower_cd = DataRescale(MaxPower_cd, *InputScaler)
    MaxPower_val = DataRescale(NN_Out,*OutputScaler)
    print("    {:7}{:7}{:7}{:11}{:9}".format('x','y','z','r','Power'))
    for coord, val in zip(MaxPower_cd,MaxPower_val[:,0]):
        print("({:.4f},{:.4f},{:.4f},{:.4f}) ---> {:.2f} W ".format(*coord, val))
    print()

    DADict["Data"]['MaxPower'] = MaxPower = {'x':MaxPower_cd[0],'y':MaxPower_val[0]}

    if ML.MaxPowerOpt.get('Verify',True):
        print("Checking results at optima\n")
        ERMESMaxPower = '{}/MaxPower.rmed'.format(DADict["CALC_DIR"])
        EMParameters = Param_ERMES(*MaxPower_cd[0],ERMES_Parameters)
        RunERMES = ML.MaxPowerOpt.get('NewSim',True)

        JH_Vol, Volumes, Elements, JH_Node = ERMES(VL,MeshFile,ERMESMaxPower,
                                                EMParameters, DADict["TMP_CALC_DIR"],
                                                RunERMES, GUI=0)
        Watts = JH_Vol*Volumes
        # Power & Uniformity
        Power = np.sum(Watts)
        JH_Node /= 1000**2
        Uniformity = UniformityScore(JH_Node,ERMESMaxPower)

        print("Anticipated power at optimum configuration: {:.2f} W".format(MaxPower_val[0,0]))
        print("Actual power at optimum configuration: {:.2f} W\n".format(Power))

        MaxPower["target"] = Power

# NetPU architecture
class NetPU(nn.Module):
    def __init__(self, Layout, Dropout):
        super(NetPU, self).__init__()
        for i, cnct in enumerate(zip(Layout,Layout[1:])):
            setattr(self,"fc{}".format(i+1),nn.Linear(*cnct))

        self.Dropout = Dropout
        self.NbLayers = len(Layout)

    def forward(self, x):
        for i, drop in enumerate(self.Dropout[:-1]):
            x = nn.Dropout(drop)(x)
            fc = getattr(self,"fc{}".format(i+1))
            x = F.leaky_relu(fc(x))

        x = nn.Dropout(self.Dropout[-1])(x)
        fc = getattr(self,'fc{}'.format(self.NbLayers-1))
        x = fc(x)
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

def NN_Opt(X,model,Ix):
    # X_tf = torch.tensor(np.atleast_2d(X),dtype=torch.float32)
    X_tf = torch.tensor(X,dtype=torch.float32)
    with torch.no_grad():
        Pred = model(X_tf).numpy()[:,Ix]

    Grad = model.Gradient(X)[:,Ix,:]

    return Pred,Grad

def Param_ERMES(x,y,z,r,Parameters):
    Parameters.update(CoilDisplacement=[x,y,z],Rotation=r)
    return Namespace(**Parameters)
