import os
import sys
from importlib import import_module, reload
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import copy
from pathos.multiprocessing import ProcessPool
import matplotlib.pyplot as plt
from matplotlib import cm
from prettytable import PrettyTable
from scipy.optimize import minimize, differential_evolution, shgo, basinhopping
from types import SimpleNamespace as Namespace
from natsort import natsorted
from Scripts.Common.VLFunctions import MeshInfo
import scipy
import pickle
from adaptive import LearnerND
import adaptive.learner.learnerND as LND


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


    # def __init__(self,bah,blah):
    #     super(NetPU, self).__init__()
    #     self.fc1 = nn.Linear(4, 32)
    #     self.fc2 = nn.Linear(32, 128)
    #     self.fc3 = nn.Linear(128, 16)
    #     self.fc4 = nn.Linear(16, 2)
    #     self.drop = nn.Dropout(0.4)
    #
    # def forward(self, x):
    #     x = nn.Dropout(0.0)(x)
    #     x = F.leaky_relu(self.fc1(x))
    #     # print(x[0,:])
    #     x = nn.Dropout(0.0)(x)
    #     x = F.leaky_relu(self.fc2(x))
    #     # print(x[0,:])
    #     x = nn.Dropout(0.4)(x)
    #     x = F.leaky_relu(self.fc3(x))
    #     # print(x[0,:])
    #     x = nn.Dropout(0.0)(x)
    #     x = self.fc4(x)
    #     print(x[0,:])
    #     return x

    def predict_denorm(self, x, xmin, xmax, ymin, ymax, GPU=False):
        # normalize
        xn = ((x[None, :] - xmin) / (xmax - xmin))[0]
        # compute
        if GPU:
            self = self.to(device)
            xn = xn.to(device)
        with torch.no_grad():
            yn = self(xn)
        if GPU:
            self = self.cpu()
            xn = xn.cpu()
            yn = yn.cpu()
        # de-normalize
        return (yn[None, :] * (ymax - ymin) + ymin)[0]

    def predict(self,xn,device=None,GPU=False):
        if GPU:
            self = self.to(device)
            xn = xn.to(device)
        with torch.no_grad():
            yn = self(xn)
        if GPU:
            self = self.cpu()
            xn = xn.cpu()
            yn = yn.cpu()
        # de-normalize
        return yn

    def GradNN(self, input):
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

def Gradold(model, input):
    # one at a time here
    # input = input.detach().numpy()
    input = np.array(input)
    for i in range(1,5):
        fc = getattr(model,'fc{}'.format(i))
        w = fc.weight.detach().numpy()
        b = fc.bias.detach().numpy()
        out = w.dot(input)+b
        diag = np.copy(out)
        diag[diag>=0] = 1
        diag[diag<0] = 0.01
        gd = diag[:,None]*w
        # print(gd.shape)
        if i==1:
            cumul = gd
        else :
            cumul = gd.dot(cumul)
        # print(cumul.shape)
        out[out<0] = out[out<0]*0.01
        input = out
    return cumul

def DataNorm(Arr, DataRange):
    return (Arr - DataRange[0])/(DataRange[1] - DataRange[0])

def DataDenorm(Arr, DataRange):
    return Arr*(DataRange[1] - DataRange[0]) + DataRange[0]

def func(X, model, alpha):
    '''
    This is the function which we look to optimise, which is a weighted geometric
    average of power & uniformity
    '''
    X = torch.tensor(X, dtype=torch.float32)
    PV = model.predict(X).detach().numpy()
    score = alpha*PV[0] + (1-alpha)*(1-PV[1])
    return -(score)

def dfunc(X, model, alpha):
    '''
    Derivative of 'func' w.r.t. the inputs x,y,z,r
    '''
    # Grad = Gradold(model, X)
    Grad = model.GradNN(X)[0]
    dscore = alpha*Grad[0,:] + (1-alpha)*(-Grad[1,:])
    return -dscore

def constraint(X, model, DesPower):
    '''
    Constraint which must be met during the optimisation of func. This specifies
    that the power must be greater than or equal to 'DesPower'
    '''
    # print(DesPower)
    X = torch.tensor(X, dtype=torch.float32)
    P,V = model.predict(X).detach().numpy()
    return (P - DesPower)

def dconstraint(X, model, DesPower):
    '''
    Derivative of constraint function
    '''
    # Grad = Gradold(model, X)
    Grad = model.GradNN(X)[0]
    dcon = Grad[0,:]
    return dcon

def MinMax(X, model, sign, Ix):
    '''
    Function which can be called to find min/max of power/uniform
    Sign = 1 => Max Value, Sign = -1 => Min Value
    Ix = 0 corresponds to power, Ix = 1 correspons to uniformity
    '''
    X = torch.tensor(X,dtype=torch.float32)
    Pred = model.predict(X).detach().numpy()[Ix]
    return -sign*Pred

def dMinMax(X, model, sign, Ix):
    '''
    Derivative of function MinMax
    '''
    # Grad = Gradold(model, X)[Ix,:]
    Grad = model.GradNN(X)[0]
    dMM = Grad[Ix,:]
    return -sign*dMM

def FuncOpt(fnc, dfnc, NbInit, bounds, **kwargs):
    Optima = []
    for X0 in np.random.uniform(0,1,size=(NbInit,4)):
        Opt = minimize(fnc, X0, jac=dfnc, method='SLSQP', bounds=bounds,**kwargs)
        if not Opt.success: continue
        Optima.append(Opt)
    return Optima

def SortOptima(Optima, tol=0.01, order='decreasing'):
    Score, Coord = [Optima[0].fun], np.array([Optima[0].x])
    for Opt in Optima[1:]:
        D = np.linalg.norm(Coord-np.array(Opt.x),axis=1)
        if all(D > tol):
            Coord = np.vstack((Coord,Opt.x))
            Score.append(Opt.fun)

    Score = np.array(Score)
    if order == 'decreasing': ord=-1
    elif order == 'increasing':ord=1
    sortlist = np.argsort(Score)[::ord]
    Score = Score[sortlist]
    Coord = Coord[sortlist,:]
    return Coord

def ErrorHist(Data,ResNames = ['Power','Uniformity']):
    bins = list(range(-20,21))
    Ranges = [5,10]
    RangeArr = [np.arange(bins.index(-i),bins.index(i)) for i in Ranges]

    NbData = Data.shape[0]
    ResData = []
    for res in Data.T:
        data = np.histogram(res,bins=bins)[0]
        ls = [100*data[arr].sum()/NbData for arr in RangeArr]
        ResData.append(ls)

    ResData = np.array(ResData)

    # Top = ['Output'] + ["+-{}%".format(v) for v in Ranges]
    # TestTb = PrettyTable(Top,float_format=".3")
    # for name, data in zip(ResNames,ResData):
    #     TestTb.add_row([name, *data])
    #
    # print(TestTb)

    return ResData

def Single(VL, MLdict):
    ML = MLdict["Parameters"]

    # File where all data is stored
    DataFile = "{}/Data.hdf5".format(VL.ML_DIR)
    # Create new data
    if ML.CreateData:
        DataDir = "{}/{}".format(VL.PROJECT_DIR,ML.DataDir)
        NewData = GetData(VL, ML, DataDir,5)
        NewData = np.array(NewData)
        MLfile = h5py.File(DataFile,'a')
        if ML.DataDir not in MLfile.keys():
            StudyGrp = MLfile.create_group(ML.DataDir)
        else :
            StudyGrp = MLfile[ML.DataDir]
        if ML.DataName in StudyGrp.keys():
            del StudyGrp[ML.DataName]
        StudyGrp.create_dataset(ML.DataName, data=NewData)
        MLfile.close()

    torch.set_num_threads(2)
    torch.manual_seed(getattr(ML,'Seed',100))

    # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')

    ModelFile = '{}/model.h5'.format(MLdict["CALC_DIR"]) # File model will be saved to/loaded from
    if ML.Train:
        # Get Training & Testing data
        MLData = h5py.File(DataFile,'r')
        if ML.TrainData in MLData: TrainData = MLData[ML.TrainData][:]
        elif "{}/{}".format(VL.StudyName,ML.TrainData) in MLData:
            TrainData = MLData["{}/{}".format(VL.StudyName,ML.TrainData)][:]
        else : sys.exit("Data not found")

        DataSplit = getattr(ML,'DataSplit',0.7)
        TrainNb = getattr(ML,'TrainNb',None)
        if not TrainNb:
            DataPrcnt = getattr(ML,'DataPrcnt',1)
            TrainNb = int(np.ceil(TrainData.shape[0]*DataPrcnt*DataSplit))

        __TrainData = TrainData[TrainNb:,:] # leftover data which may be used for testing
        TrainData = TrainData[:TrainNb,:] # Data used for training

        TestNb = int(np.ceil(TrainNb*(1-DataSplit)/DataSplit))
        if hasattr(ML,'TestData'):
            if ML.TestData == ML.TrainData: TestData = __TrainData
            elif ML.TestData in MLData : TestData = MLData[ML.TestData][:]
            else : sys.exit('No testing data')
        else:
            TestData = __TrainData

        TestData = TestData[:TestNb,:]
        MLData.close()

        # Convert data to float32 for PyTorch
        TrainData = TrainData.astype('float32')
        TestData = TestData.astype('float32')

        # Normalise input training data to [0,1] range to speed up convergence.
        # Nornalise output to [0,1] ensures both are rated equally during backprop
        DataMin, DataMax = TrainData.min(axis=0), TrainData.max(axis=0)
        DataRange = np.array([DataMin,DataMax])
        InputRange, OutputRange  = DataRange[:,:4],DataRange[:,4:]

        TrainData_norm = DataNorm(TrainData,DataRange)
        In_train = torch.from_numpy(TrainData_norm[:,:4])
        Out_train = torch.from_numpy(TrainData_norm[:,4:])
        # Scale test data by train data range
        TestData_norm = DataNorm(TestData,DataRange)
        In_test  = torch.from_numpy(TestData_norm[:,:4])
        Out_test  = torch.from_numpy(TestData_norm[:,4:])

        GPU = getattr(ML,'GPU', False)
        lr = getattr(ML,'lr', 0.0001)
        PrintEpoch = getattr(ML,'PrintEpoch',100)
        ConvCheck = getattr(ML,'ConvCheck',100)

        model = NetPU(ML.NNLayout,ML.Dropout)
        model.train()

        # Create batches of the data
        train_dataset = Data.TensorDataset(In_train, Out_train)
        train_loader = Data.DataLoader(train_dataset, batch_size=ML.BatchSize, shuffle=True)

        # train on GPU
        if GPU:
            model = model.to(device)
            In_test, Out_test = In_test.to(device), Out_test.to(device)

        # optimizer
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        # loss tensor
        # IndLoss = nn.MSELoss(reduction='none')
        loss_func = nn.MSELoss(reduction='mean')
        # Convergence history
        LossConv = {'loss_batch': [], 'loss_train': [], 'loss_test': []}

        BestModel = copy.deepcopy(model)
        BestLoss_test, OldAvg = float('inf'), float('inf')
        print("Starting training")
        print("Training set: {}\nTest set: {}\n".format(In_train.size()[0], In_test.size()[0]))
        print("{:8}{:12}{:12}".format('Epoch','Train_loss','Val_loss'))
        for epoch in np.arange(1,ML.NbEpoch+1):
            # Loop through the batches for each epoch
            for step, (batch_x, batch_y) in enumerate(train_loader):
                if GPU:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                # forward and loss
                loss = loss_func(model(batch_x), batch_y)
                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # validate
            model.eval() # Change to eval to switch off gardients and dropout
            loss_test = loss_func(model(In_test), Out_test)
            loss_train = loss_func(model(In_train), Out_train)
            model.train()

            LossConv['loss_batch'].append(loss.cpu().detach().numpy().tolist()) #loss of final batch
            LossConv['loss_test'].append(loss_test.cpu().detach().numpy().tolist()) # loss of full test
            LossConv['loss_train'].append(loss_train.cpu().detach().numpy().tolist()) #loss of full train

            # loss_val_sep = torch.mean(IndLoss(model(test_rxyz_norm), test_mu_sigma_norm),dim=0)
            # loss_train_sep = torch.mean(IndLoss(model(train_rxyz_norm), train_mu_sigma_norm),dim=0)

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
                    break
                OldAvg = Avg

        model = BestModel
        if GPU:
            model = model.to(device)
            In_test, Out_test = In_test.cpu(), Out_test.cpu()

        print('Training complete\n')
        print("Training loss: {:.8f}\nValidation loss: {:.8f}".format(BestLoss_train,BestLoss_test))

        # save model & training/testing data
        torch.save(model.state_dict(), ModelFile)
        np.save("{}/TrainData".format(MLdict["CALC_DIR"]),TrainData)
        np.save("{}/TestData".format(MLdict["CALC_DIR"]),TestData)

        plt.figure(figsize=(12,9),dpi=100)
        plt.plot(LossConv['loss_train'][0:epoch], label='Train')
        plt.plot(LossConv['loss_test'][0:epoch], label='Test')
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.legend()
        plt.savefig("{}/Convergence.png".format(MLdict["CALC_DIR"]))
        plt.close()

        # Predicted values from test and train data
        with torch.no_grad():
            NN_train = model.predict(In_train)
            NN_test = model.predict(In_test)
        NN_train = DataDenorm(NN_train, OutputRange)
        NN_test = DataDenorm(NN_test, OutputRange)

        # Error percentage for predicted test and train
        train_error = (NN_train - TrainData[:,4:]).detach().numpy()
        train_error_perc = 100*train_error/TrainData[:,4:]
        test_error = (NN_test - TestData[:,4:]).detach().numpy()
        test_error_perc = 100*test_error/TestData[:,4:]

        print("\nTraining error percentages")
        TrainHist = ErrorHist(train_error_perc)

        print("\nTesting error percentages")
        TestHist = ErrorHist(test_error_perc)

        bins = list(range(-20,21))
        Xlim = 10

        fig, axes = plt.subplots(1, 2,figsize=(16,8))
        fig.suptitle('Test data')
        for ax, res, name in zip(axes, test_error_perc.T, ['Power','Uniformity']):
            ax.hist(res,bins)
            ax.set_title(name)
            ax.set_xlabel('Error')
            ax.set_ylabel('Count')
            Xlim = max(Xlim,*np.absolute(ax.get_xlim()))
        plt.setp(ax, xlim=(-Xlim,Xlim))
        plt.savefig("{}/TestLossHistogram.png".format(MLdict["CALC_DIR"]))
        plt.close()

        fig, axes = plt.subplots(1, 2,figsize=(16,8))
        fig.suptitle('Train data')
        for ax, res, name in zip(axes, train_error_perc.T, ['Power','Uniformity']):
            ax.hist(res,bins)
            ax.set_title(name)
            ax.set_xlabel('Error')
            ax.set_ylabel('Count')
        plt.setp(ax, xlim=(-Xlim,Xlim))
        plt.savefig("{}/TrainLossHistogram.png".format(MLdict["CALC_DIR"]))
        plt.close()

        Slice = 'xy'
        Res = 'Power' # 'Uniformity'
        Component = 'val'
        MajorN, MinorN = 7, 20

        InTag, OutTag = ['x','y','z','r'],['power','uniformity']

        AxMin, AxMaj = [], []
        for i, cmp in enumerate(InTag):
            if Slice.find(cmp) == -1: AxMaj.append(i)
            else : AxMin.append(i)

        ResAx = OutTag.index(Res.lower())
        # Discretisation
        DiscMin = np.linspace(0+0.5*1/MinorN,1-0.5*1/MinorN,MinorN)
        DiscMaj = np.linspace(0,1,MajorN)
        # DiscMaj = np.linspace(0+0.5*1/MajorN,1-0.5*1/MajorN,MajorN)
        disc = [DiscMaj]*4
        disc[AxMin[0]] = disc[AxMin[1]] = DiscMin
        grid = np.meshgrid(*disc, indexing='ij')
        grid = np.moveaxis(np.array(grid),0,-1) #grid point is now the last axis
        ndim = grid.ndim - 1
        # unroll grid so it can be passed to model
        _disc = [dsc.shape[0] for dsc in disc]
        grid = grid.reshape([np.prod(_disc),ndim])

        # decide what component of Res to print - gradient (magnitude or individual) or the value
        if Component.lower().startswith('grad'):
            imres = model.GradNN(grid)
            if len(Component) == 4:
                imres = np.linalg.norm(imres, axis=2)
            else:
                # letter after grad must be x,y,z or r
                ax = InTag.index(Component[-1].lower())
                imres = imres[:,:,ax]
        else :
            with torch.no_grad():
                imres = model.predict(torch.tensor(grid, dtype=torch.float32)).detach().numpy()
                imres = DataDenorm(imres,OutputRange)

        # min and max value for global colour bar
        IMMin, IMMax = imres.min(axis=0), imres.max(axis=0)
        imres = imres.reshape(_disc+[*imres.shape[1:]])
        # scaled vales for major axis
        arr1 = DataDenorm(DiscMaj,InputRange[:,AxMaj[0]])
        arr2 = DataDenorm(DiscMaj,InputRange[:,AxMaj[1]])

        PlotTrain=False
        df = DiscMaj[1] - DiscMaj[0]
        errmin, errmax = train_error.min(axis=0)[ResAx], train_error.max(axis=0)[ResAx]
        errbnd = max(abs(errmin),abs(errmax))

        fig, ax = plt.subplots(nrows=arr1.size, ncols=arr2.size, sharex=True, sharey=True, dpi=200, figsize=(12,9))
        ax = np.atleast_2d(ax)
        fig.subplots_adjust(right=0.8)
        for it1,(dsc1,nb1) in enumerate(zip(DiscMaj,arr1)):
            _it1 = -(it1+1)
            ax[_it1, 0].set_ylabel('{:.4f}'.format(nb1), fontsize=12)
            for it2,(dsc2,nb2) in enumerate(zip(DiscMaj,arr2)):
                sl = [slice(None)]*len(InTag) + [ResAx]
                sl[AxMaj[0]],sl[AxMaj[1]]  = it1, it2

                Im = ax[_it1,it2].imshow(imres[tuple(sl)].T, cmap = 'coolwarm', vmin=IMMin[ResAx], vmax=IMMax[ResAx], origin='lower',extent=(0,1,0,1))
                ax[_it1,it2].set_xticks([])
                ax[_it1,it2].set_yticks([])

                ax[-1, it2].set_xlabel("{:.4f}".format(nb2), fontsize=12)
                if PlotTrain:
                    limmin = TrainData_norm[:,AxMaj] > [dsc1-0.5*df,dsc2-0.5*df]
                    limmax = TrainData_norm[:,AxMaj] < [dsc1+0.5*df,dsc2+0.5*df]
                    bl = limmin[:,0]*limmin[:,1]*limmax[:,0]*limmax[:,1]
                    dat = TrainData_norm[bl][:,AxMin]
                    cl = train_error[:,ResAx][bl]
                    cmap = cm.get_cmap('PiYG')
                    Sc = ax[_it1,it2].scatter(*dat.T, c=cl, cmap=cmap, marker = 'x', vmin = -errbnd, vmax = errbnd, s=2)

        if PlotTrain:
            cbar_ax = fig.add_axes([0.85, 0.05, 0.05, 0.4])
            fig.colorbar(Sc, cax=cbar_ax)
            cbar_ax = fig.add_axes([0.85, 0.55, 0.05, 0.4])
        else :
            cbar_ax = fig.add_axes([0.85, 0.2, 0.05, 0.6])

        bnds = np.linspace(*Im.get_clim(),12)
        ticks = np.linspace(*Im.get_clim(),6)
        fig.colorbar(Im, boundaries=bnds, ticks=ticks, cax=cbar_ax)

        fig.suptitle(Res.capitalize())

        fig.text(0.04, 0.5, InTag[AxMaj[0]].capitalize(), ha='center')
        fig.text(0.5, 0.04, InTag[AxMaj[1]].capitalize(), va='center', rotation='vertical')
        plt.savefig("{}/NNPlot.png".format(MLdict["CALC_DIR"]))

    else:
        model = NetPU(ML.NNLayout,ML.Dropout)
        model.load_state_dict(torch.load(ModelFile))

        TrainData = np.load("{}/TrainData.npy".format(MLdict["CALC_DIR"]))
        DataMin, DataMax = TrainData.min(axis=0), TrainData.max(axis=0)
        DataRange = np.array([DataMin,DataMax])
        InputRange, OutputRange  = DataRange[:,:4],DataRange[:,4:]
        TestData = np.load("{}/TestData.npy".format(MLdict["CALC_DIR"]))

    model.eval()

    # TrainData_norm = DataNorm(TrainData,DataRange)
    # TestData_norm = DataNorm(TestData,DataRange)
    # In_train = torch.from_numpy(TrainData_norm[:,:4])
    # Out_train = torch.from_numpy(TrainData_norm[:,4:])
    # In_test = torch.from_numpy(TestData_norm[:,:4])
    # Out_test = torch.from_numpy(TestData_norm[:,4:])
    #
    # trainloss = nn.MSELoss(reduction='mean')(model(In_train),Out_train)
    # testloss = nn.MSELoss(reduction='mean')(model(In_test),Out_test)
    #
    # MLdict["TrainLoss"] = float(trainloss.detach().numpy())
    # MLdict["TestLoss"] = float(testloss.detach().numpy())

    return
    # Find the point(s) which give the maximum power
    b = (0.0,1.0)
    bnds = (b, b, b, b)

    # Get max point in NN
    Optima = FuncOpt(MinMax,dMinMax,20,bnds,args=(model,1,0))
    MaxPower_cd = SortOptima(Optima, order='increasing')
    with torch.no_grad():
        NNout = model.predict(torch.tensor(MaxPower_cd, dtype=torch.float32))
    MaxPower_val = DataDenorm(NNout,OutputRange)
    MaxPower_cd = DataDenorm(MaxPower_cd, InputRange)
    print("Optimum configuration(s) for max. power:")
    print("{:8}{:12}{:12}".format('Epoch','Train_loss','Val_loss'))
    print("    {:7}{:7}{:7}{:11}{:9}{:8}".format('x','y','z','r','Power','Variation'))
    for coord, val in zip(MaxPower_cd,MaxPower_val):
        print("({:.4f},{:.4f},{:.4f},{:.4f}) ---> {:.2f} W, {:.3f}".format(*coord, *val))
    print()

    # Find optimum uniformity for a given power
    if hasattr(ML,'DesiredPower'):
        if ML.DesiredPower > MaxPower_val[0,0]:
            print('DesiredPower greater than power available')
            DesPower = None
        else:
            DesPower = ML.DesiredPower
    else: DesPower=None

    if DesPower != None:
        DesPower_norm = DataNorm(np.array([DesPower,0]), OutputRange)[0]
        con1 = {'type': 'ineq', 'fun': constraint,'jac':dconstraint, 'args':(model, DesPower_norm)}

        Optima = FuncOpt(MinMax,dMinMax,10,bnds,args=(model,-1,1),constraints=con1,options={'maxiter':100})
        OptUni_cd = SortOptima(Optima, order='increasing')
        with torch.no_grad():
            NNout = model.predict(torch.tensor(OptUni_cd, dtype=torch.float32))
        OptUni_val = DataDenorm(NNout,OutputRange)
        OptUni_cd = DataDenorm(OptUni_cd, InputRange)

        print("Optimum configuration(s) for max. uniformity  (ensuring power >= {} W):".format(DesPower))
        print("{:8}{:12}{:12}".format('Epoch','Train_loss','Val_loss'))
        print("    {:7}{:7}{:7}{:11}{:9}{:8}".format('x','y','z','r','Power','Variation'))
        for coord, val in zip(OptUni_cd,OptUni_val):
            print("({:.4f},{:.4f},{:.4f},{:.4f}) ---> {:.2f} W, {:.3f}".format(*coord, *val))
        print()

    if hasattr(ML,'Alpha'):
        if ML.Alpha < 0 or ML.Alpha >1:
            print('Alpha must be between 0 and 1')
            alpha = None
        else : alpha = ML.Alpha
    else: alpha = None
    if alpha != None:
        Optima = FuncOpt(func,dfunc,10,bnds,args=(model,alpha))
        W_avg_cd = SortOptima(Optima, order='increasing')
        with torch.no_grad():
            NNout = model.predict(torch.tensor(W_avg_cd, dtype=torch.float32))
        W_avg_val = DataDenorm(NNout,OutputRange)
        W_avg_cd = DataDenorm(W_avg_cd, InputRange)
        print("Optimum configuration(s) for weighted average (alpha = {}):".format(alpha))
        print("{:8}{:12}{:12}".format('Epoch','Train_loss','Val_loss'))
        print("    {:7}{:7}{:7}{:11}{:9}{:8}".format('x','y','z','r','Power','Variation'))
        for coord, val in zip(W_avg_cd,W_avg_val):
            print("({:.4f},{:.4f},{:.4f},{:.4f}) ---> {:.2f} W, {:.3f}".format(*coord, *val))
        print()

    return

    NbInit = 5
    rnd = np.random.uniform(0,1,size=(NbInit,4))
    OptScores = []
    for i, X0 in enumerate(rnd):
        OptScore = minimize(func, X0, args=(model, alpha), method='SLSQP',jac=dfunc, bounds=bnds, constraints=cnstr, options={'maxiter':100})
        if OptScore.success: OptScores.append(OptScore)

    Score = []
    tol = 0.001
    for Opt in OptScores:
        if not Score:
            Score, Coord = [-Opt.fun], np.array([Opt.x])
        else :
            D = np.linalg.norm(Coord-np.array(Opt.x),axis=1)
            # print(D.min())
            # bl = D < tol
            # if any(bl):
            #     print(Opt.x,Coord[bl,:])
            if all(D > tol):
                Coord = np.vstack((Coord,Opt.x))
                Score.append(-Opt.fun)

    Score = np.array(Score)
    # print(Score, Coord)
    sortlist = np.argsort(Score)[::-1]
    Score = Score[sortlist]
    Coord = Coord[sortlist,:]

    NNOptOutput = model.predict(torch.tensor(Coord, dtype=torch.float32))
    NNOptOutput = (NNOptOutput*(OutputRange[1]-OutputRange[0]) + OutputRange[0]).detach().numpy()
    OptCoord = Coord*(InputRange[1]-InputRange[0]) + InputRange[0]
    BestCoord, BestPred = OptCoord[0,:], NNOptOutput[0,:]
    print("Optimum configuration:")
    print("x,y,z,r ---> ({:.4f},{:.4f},{:.4f},{:.4f})".format(*BestCoord))
    print("Power, Uniformity ---> {:.2f} W, {:.3f}\n".format(*BestPred))
    print()

    if OptCoord.shape[0]>1:
        Nb = 5 # Max number of other configurations to show
        print("Other configurations:")
        for Cd, Pred in zip(OptCoord[1:Nb+1,:],NNOptOutput[1:,:]):
            print("x,y,z,r ---> ({:.4f},{:.4f},{:.4f},{:.4f})".format(*Cd))
            print("Power, Uniformity ---> {:.2f} W, {:.3f}\n".format(*Pred))


    # shgo doesn't seem to work as well with constraints - often numbers lower than des power are generated as optimum
    # This is because of max iter in solver however no return of success for each run with shgo
    # OptScore = shgo(func, bnds, args=(model,alpha), n=100, iters=5, sampling_method='sobol',minimizer_kwargs={'method':'SLSQP','jac':dfunc,'args':(model, alpha),'constraints':cnstr})
    # print(OptScore)

    if ML.Verify:
        from PreAster.devPreHIVE import ERMES_Mesh, SetupERMES

        ParaDict = {'CoilType':'HIVE',
                    'CoilDisplacement':BestCoord[:3].tolist(),
                    'Rotation':BestCoord[3],
                    'Current':1000,
                    'Frequency':1e4,
                    'Materials':{'Block':'Copper_NL', 'Pipe':'Copper_NL', 'Tile':'Tungsten_NL'}}

        VL.WriteModule("{}/Parameters.py".format(VL.tmpML_DIR), ParaDict)

        Parameters = Namespace(**ParaDict)

        ERMESresfile = '{}/maxERMES.rmed'.format(MLdict["CALC_DIR"])
        SampleMeshFile = "{}/AMAZEsample.med".format(VL.MESH_DIR)

        if ML.NewSim:
            ERMESdir = "{}/ERMES".format(VL.tmpML_DIR)
            os.makedirs(ERMESdir)
            # Create ERMES mesh
            ERMESmeshfile = "{}/Mesh.med".format(ERMESdir)
            err = ERMES_Mesh(VL,SampleMeshFile, ERMESmeshfile,
            				AddPath = VL.tmpML_DIR,
            				LogFile = None,
            				GUI=0)
            if err: return sys.exit('Issue creating mesh')

            Watts, WattsPV, Elements, JHNode = SetupERMES(VL, Parameters, ERMESmeshfile, ERMESresfile, ERMESdir)

            # shutil.rmtree(ERMESdir)
        else :
            ERMESres = h5py.File(ERMESresfile, 'r')
            attrs =  ERMESres["EM_Load"].attrs
            Elements = ERMESres["EM_Load/Elements"][:]

            Scale = (Parameters.Current/attrs['Current'])**2
            Watts = ERMESres["EM_Load/Watts"][:]*Scale
            WattsPV = ERMESres["EM_Load/WattsPV"][:]*Scale
            JHNode =  ERMESres["EM_Load/JHNode"][:]*Scale
            ERMESres.close()

        # Power
        Power = np.sum(Watts)
        # Uniformity

        JHNode /= Parameters.Current**2
        Meshcls = MeshInfo(SampleMeshFile)
        CoilFace = Meshcls.GroupInfo('CoilFace')
        Area, JHArea = 0, 0 # Actual area and area of triangles with JH
        for nodes in CoilFace.Connect:
            vertices = Meshcls.GetNodeXYZ(nodes)
            # Heron's formula
            a, b, c = scipy.spatial.distance.pdist(vertices, metric="euclidean")
            s = 0.5 * (a + b + c)
            area = np.sqrt(s * (s - a) * (s - b) * (s - c))
            Area += area

            vertices[:,2] += JHNode[nodes - 1].flatten()

            a, b, c = scipy.spatial.distance.pdist(vertices, metric="euclidean")
            s = 0.5 * (a + b + c)
            area1 = np.sqrt(s * (s - a) * (s - b) * (s - c))
            JHArea += area1

        Meshcls.Close()
        Uniformity = JHArea/Area

        ActOptOutput = np.array([Power, Uniformity])

        print("Actual power & uniformity at optimum configuration is {:.2f} W, {:.3f}".format(*ActOptOutput))

        err = 100*(BestPred - ActOptOutput)/ActOptOutput
        print("Prediction errors are: {:.3f} & {:.3f}".format(*err))


# def Combined(VL,MLDicts):
#     TrainDataDict = {}
#     TestDataDict = {}
#     for MLdict in MLDicts:
#         Nb = MLdict["Parameters"].TrainNb
#         if Nb not in TrainDataDict: TrainDataDict[Nb] = []
#         if Nb not in TestDataDict: TestDataDict[Nb] = []
#
#         TrainDataDict[Nb].append(MLdict["TrainLoss"])
#         TestDataDict[Nb].append(MLdict["TestLoss"])
#
#     plt.figure(figsize=(12,9))
#     AvgTest, AvgTrain = [],[]
#     ErrTest, ErrTrain = [],[]
#     Nbs = list(TestDataDict.keys())
#     for Nb in Nbs:
#         TestVal = TestDataDict[Nb]
#         _AvgTest = np.mean(TestVal)
#         AvgTest.append(_AvgTest)
#         ErrTest.append([_AvgTest-np.min(TestVal),np.max(TestVal)-_AvgTest])
#
#         TrainVal = TrainDataDict[Nb]
#         _AvgTrain = np.mean(TrainVal)
#         AvgTrain.append(_AvgTrain)
#         ErrTrain.append([_AvgTrain-np.min(TrainVal),np.max(TrainVal)-_AvgTrain])
#
#     plt.errorbar(Nbs,AvgTest,yerr=np.array(ErrTest).T,fmt='bo',ecolor='b',capsize=5,label='Test')
#     plt.errorbar(Nbs,AvgTrain,yerr=np.array(ErrTrain).T,fmt='go',ecolor='g',capsize=5,label='Train')
#
#     plt.xlabel('TrainNb')
#     plt.ylabel('MSE')
#     plt.legend()
#     plt.show()
#     # plt.savefig("{}/Methods/NNerr.png".format(VL.ML_DIR,))
#     plt.close()
#     # plt.figure(figsize=(12,9))
#     # for Nb, vals in TestDataDict.items():
#     #     x = [Nb]*len(vals)
#     #     plt.scatter(x,vals)
#     # plt.savefig("{}/Methods/TestData.png".format(VL.ML_DIR,))
#     # plt.close()
#     #
#     # plt.figure(figsize=(12,9))
#     # for Nb, vals in TrainDataDict.items():
#     #     x = [Nb]*len(vals)
#     #     plt.scatter(x,vals)
#     # plt.savefig("{}/Methods/TrainData.png".format(VL.ML_DIR,))
#     # plt.close()
