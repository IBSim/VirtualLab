import os
import sys
import h5py
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib import cm
from prettytable import PrettyTable
from scipy.optimize import minimize, differential_evolution, shgo, basinhopping
from types import SimpleNamespace as Namespace
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data

from Scripts.Common.VLFunctions import MeshInfo

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

    def predict_denorm(self, x, InputRange, OutputRange, GPU=False):
        # normalize
        xn = (x - InputRange[0])/(InputRange[1] - InputRange[0])
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
        return (yn[None, :] * (OutputRange[1] - OutputRange[0]) + OutputRange[0])

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

    return ResData

def VerifyNN(VL,DataDict):
    from PreAster.devPreHIVE import ERMES_Mesh, SetupERMES

    os.makedirs(DataDict['ERMESdir'],exist_ok=True)

    # Create ERMES mesh
    err = ERMES_Mesh(VL,DataDict)
    if err: return sys.exit('Issue creating mesh')

    ERMESres = SetupERMES(VL, DataDict['Parameters'],
                                       DataDict['OutputFile'],
                                       DataDict['ERMESResFile'], DataDict['ERMESdir'])
    return ERMESres



def Single(VL, MLdict):
    ML = MLdict["Parameters"]

    # File where all data is stored
    DataFile = "{}/Data.hdf5".format(VL.ML_DIR)
    NbTorchThread = getattr(ML,'NbTorchThread',1)
    torch.set_num_threads(NbTorchThread)
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
        plt.close()
    else:
        model = NetPU(ML.NNLayout,ML.Dropout)
        model.load_state_dict(torch.load(ModelFile))

        TrainData = np.load("{}/TrainData.npy".format(MLdict["CALC_DIR"]))
        DataMin, DataMax = TrainData.min(axis=0), TrainData.max(axis=0)
        DataRange = np.array([DataMin,DataMax])
        InputRange, OutputRange  = DataRange[:,:4],DataRange[:,4:]
        TestData = np.load("{}/TestData.npy".format(MLdict["CALC_DIR"]))

    model.eval()

    Check = getattr(ML,'Check',None)
    if Check:
        x = torch.tensor(Check, dtype=torch.float32)
        y = model.predict_denorm(x, InputRange, OutputRange)
        MLdict['CheckAns'] = y.detach().numpy().tolist()

    # set bounds for optimsation
    b = (0.0,1.0)
    bnds = (b, b, b, b)

    # Optimsation 1: Find the point of max power
    # Find the point(s) which give the maximum power
    if hasattr(ML,'MaxPowerOpt'):
        print("Optimum configuration(s) for max. power")
        NbInit = ML.MaxPowerOpt.get('NbInit',20)
        # Get max point in NN. Create NbInit random seeds to start
        Optima = FuncOpt(MinMax,dMinMax,NbInit,bnds,args=(model,1,0))
        MaxPower_cd = SortOptima(Optima, tol=0.05, order='increasing')
        with torch.no_grad():
            NNout = model.predict(torch.tensor(MaxPower_cd, dtype=torch.float32))
        MaxPower_val = DataDenorm(NNout,OutputRange).detach().numpy()
        MaxPower_cd = DataDenorm(MaxPower_cd, InputRange)
        print("{:8}{:12}{:12}".format('Epoch','Train_loss','Val_loss'))
        print("    {:7}{:7}{:7}{:11}{:9}{:8}".format('x','y','z','r','Power','Variation'))
        for coord, val in zip(MaxPower_cd,MaxPower_val):
            print("({:.4f},{:.4f},{:.4f},{:.4f}) ---> {:.2f} W, {:.3f}".format(*coord, *val))
        print()

        MLdict['Optima_1'] = np.hstack((MaxPower_cd, MaxPower_val)).tolist()

        if ML.MaxPowerOpt.get('Verify',True):
            CheckPoint = MaxPower_cd[0].tolist()
            print("Checking results at {}\n".format(CheckPoint))

            ERMESResFile = '{}/MaxPower.rmed'.format(MLdict["CALC_DIR"])
            if ML.MaxPowerOpt.get('NewSim',True):
                ParaDict = {'CoilType':'HIVE',
                            'CoilDisplacement':CheckPoint[:3],
                            'Rotation':CheckPoint[3],
                            'Current':1000,
                            'Frequency':1e4,
                            'Materials':{'Block':'Copper_NL', 'Pipe':'Copper_NL', 'Tile':'Tungsten_NL'}}
                Parameters = Namespace(**ParaDict)
                ERMESdir = "{}/ERMES".format(MLdict['TMP_CALC_DIR'])
                DataDict = {'InputFile':"{}/AMAZEsample.med".format(VL.MESH_DIR),
                            'OutputFile':"{}/Mesh.med".format(ERMESdir),
                            'ERMESResFile':ERMESResFile,
                            'ERMESdir':ERMESdir,
                            'Parameters':Parameters}

                Watts, WattsPV, Elements, JHNode = VerifyNN(VL, DataDict)
            elif os.path.isfile(ERMESResFile):
                ERMESres = h5py.File(ERMESResFile, 'r')
                attrs =  ERMESres["EM_Load"].attrs
                Elements = ERMESres["EM_Load/Elements"][:]

                Scale = (1000/attrs['Current'])**2
                Watts = ERMESres["EM_Load/Watts"][:]*Scale
                WattsPV = ERMESres["EM_Load/WattsPV"][:]*Scale
                JHNode =  ERMESres["EM_Load/JHNode"][:]*Scale
                ERMESres.close()

            # Power & Uniformity
            Power = np.sum(Watts)
            JHNode /= 1000**2
            Uniformity = UniformityScore(JHNode,ERMESResFile)

            print("Anticipated power & uniformity at optimum configuration is {:.2f} W, {:.3f}".format(*MaxPower_val[0]))
            print("Actual power & uniformity at optimum configuration is {:.2f} W, {:.3f}\n".format(Power,Uniformity))

            # err = 100*(MaxPower_val[0,:] - ActOptOutput)/ActOptOutput
            # print("Prediction errors are: {:.3f} & {:.3f}".format(*err))
            # MLdict['Optima'] = [MaxPower_val[0,0],Power]

    #Optimisation2: Find optimum uniformity for a given power
    if hasattr(ML,'DesPowerOpt'):
        if ML.DesPowerOpt['Power'] >= MaxPower_val[0,0]:
            print('DesiredPower greater than power available.\n')
        else:
            print("Optimum configuration(s) for max. uniformity  (ensuring power >= {} W)".format(ML.DesPowerOpt['Power']))
            DesPower_norm = DataNorm(np.array([ML.DesPowerOpt['Power'],0]), OutputRange)[0]

            NbInit = ML.DesPowerOpt.get('NbInit',20)
            # constraint to ensure des power is met
            con1 = {'type': 'ineq', 'fun': constraint,'jac':dconstraint, 'args':(model, DesPower_norm)}
            Optima = FuncOpt(MinMax,dMinMax,NbInit,bnds,args=(model,-1,1),constraints=con1,options={'maxiter':100})
            OptUni_cd = SortOptima(Optima, order='increasing')
            with torch.no_grad():
                NNout = model.predict(torch.tensor(OptUni_cd, dtype=torch.float32))
            OptUni_val = DataDenorm(NNout,OutputRange).detach().numpy()
            OptUni_cd = DataDenorm(OptUni_cd, InputRange)

            print("{:8}{:12}{:12}".format('Epoch','Train_loss','Val_loss'))
            print("    {:7}{:7}{:7}{:11}{:9}{:8}".format('x','y','z','r','Power','Variation'))
            for coord, val in zip(OptUni_cd,OptUni_val):
                print("({:.4f},{:.4f},{:.4f},{:.4f}) ---> {:.2f} W, {:.3f}".format(*coord, *val))
            print()

            if ML.DesPowerOpt.get('Verify',True):
                CheckPoint = OptUni_cd[0].tolist()
                print("Checking results at {}\n".format(CheckPoint))

                ERMESResFile = '{}/DesPower.rmed'.format(MLdict["CALC_DIR"])
                if ML.DesPowerOpt.get('NewSim',True):
                    ParaDict = {'CoilType':'HIVE',
                                'CoilDisplacement':CheckPoint[:3],
                                'Rotation':CheckPoint[3],
                                'Current':1000,
                                'Frequency':1e4,
                                'Materials':{'Block':'Copper_NL', 'Pipe':'Copper_NL', 'Tile':'Tungsten_NL'}}
                    Parameters = Namespace(**ParaDict)
                    ERMESdir = "{}/ERMES".format(MLdict['TMP_CALC_DIR'])
                    DataDict = {'InputFile':"{}/AMAZEsample.med".format(VL.MESH_DIR),
                                'OutputFile':"{}/Mesh.med".format(ERMESdir),
                                'ERMESResFile':ERMESResFile,
                                'ERMESdir':ERMESdir,
                                'Parameters':Parameters}

                    Watts, WattsPV, Elements, JHNode = VerifyNN(VL, DataDict)
                elif os.path.isfile(ERMESResFile):
                    ERMESres = h5py.File(ERMESResFile, 'r')
                    attrs =  ERMESres["EM_Load"].attrs
                    Elements = ERMESres["EM_Load/Elements"][:]

                    Scale = (1000/attrs['Current'])**2
                    Watts = ERMESres["EM_Load/Watts"][:]*Scale
                    WattsPV = ERMESres["EM_Load/WattsPV"][:]*Scale
                    JHNode =  ERMESres["EM_Load/JHNode"][:]*Scale
                    ERMESres.close()

                # Power & Uniformity
                Power = np.sum(Watts)
                JHNode /= 1000**2
                Uniformity = UniformityScore(JHNode,ERMESResFile)
                print("Anticipated power & uniformity at optimum configuration is {:.2f} W, {:.3f}".format(*OptUni_val[0]))
                print("Actual power & uniformity at optimum configuration is {:.2f} W, {:.3f}\n".format(Power,Uniformity))

    # Optimsation 3: Weighted average of Power & Uniformity
    if hasattr(ML,'CombinedOpt'):
        print("Optimum configuration(s) for weighted average (alpha = {})".format(ML.CombinedOpt['Alpha']))
        NbInit = ML.CombinedOpt.get('NbInit',20)
        Optima = FuncOpt(func,dfunc,10,bnds,args=(model,ML.CombinedOpt['Alpha']))
        W_avg_cd = SortOptima(Optima, order='increasing')
        with torch.no_grad():
            NNout = model.predict(torch.tensor(W_avg_cd, dtype=torch.float32))
        W_avg_val = DataDenorm(NNout,OutputRange).detach().numpy()
        W_avg_cd = DataDenorm(W_avg_cd, InputRange)

        print("{:8}{:12}{:12}".format('Epoch','Train_loss','Val_loss'))
        print("    {:7}{:7}{:7}{:11}{:9}{:8}".format('x','y','z','r','Power','Variation'))
        for coord, val in zip(W_avg_cd,W_avg_val):
            print("({:.4f},{:.4f},{:.4f},{:.4f}) ---> {:.2f} W, {:.3f}".format(*coord, *val))
        print()

        if ML.CombinedOpt.get('Verify',True):
            CheckPoint = W_avg_cd[0].tolist()
            print("Checking results at {}\n".format(CheckPoint))

            ERMESResFile = '{}/WeightedAverage.rmed'.format(MLdict["CALC_DIR"])
            if ML.CombinedOpt.get('NewSim',True):
                ParaDict = {'CoilType':'HIVE',
                            'CoilDisplacement':CheckPoint[:3],
                            'Rotation':CheckPoint[3],
                            'Current':1000,
                            'Frequency':1e4,
                            'Materials':{'Block':'Copper_NL', 'Pipe':'Copper_NL', 'Tile':'Tungsten_NL'}}
                Parameters = Namespace(**ParaDict)
                ERMESdir = "{}/ERMES".format(MLdict['TMP_CALC_DIR'])
                DataDict = {'InputFile':"{}/AMAZEsample.med".format(VL.MESH_DIR),
                            'OutputFile':"{}/Mesh.med".format(ERMESdir),
                            'ERMESResFile':ERMESResFile,
                            'ERMESdir':ERMESdir,
                            'Parameters':Parameters}

                Watts, WattsPV, Elements, JHNode = VerifyNN(VL, DataDict)
            elif os.path.isfile(ERMESResFile):
                ERMESres = h5py.File(ERMESResFile, 'r')
                attrs =  ERMESres["EM_Load"].attrs
                Elements = ERMESres["EM_Load/Elements"][:]

                Scale = (1000/attrs['Current'])**2
                Watts = ERMESres["EM_Load/Watts"][:]*Scale
                WattsPV = ERMESres["EM_Load/WattsPV"][:]*Scale
                JHNode =  ERMESres["EM_Load/JHNode"][:]*Scale
                ERMESres.close()

            # Power & Uniformity
            Power = np.sum(Watts)
            JHNode /= 1000**2
            Uniformity = UniformityScore(JHNode,ERMESResFile)
            print("Anticipated power & uniformity at optimum configuration is {:.2f} W, {:.3f}".format(*W_avg_val[0]))
            print("Actual power & uniformity at optimum configuration is {:.2f} W, {:.3f}\n".format(Power,Uniformity))

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

# def Combined(VL,MLDicts):
#     DataDict = {}
#     for MLdict in MLDicts:
#         Name = MLdict['Parameters'].TrainData.split('/')[0]
#         Nb = MLdict['Parameters'].TrainNb
#         if Name not in DataDict: DataDict[Name] = {}
#         if Nb not in DataDict[Name]: DataDict[Name][Nb] = []
#         DataDict[Name][Nb].append(MLdict['Optima'][1])
#
#     plt.figure(figsize=(12,9))
#     for Name, _dict in DataDict.items():
#         datx,daty = [],[]
#         for Nb, vals in _dict.items():
#             datx+=[Nb]*len(vals)
#             daty+=vals
#         plt.scatter(datx,daty, label=Name)
#     plt.legend()
#     plt.show()
