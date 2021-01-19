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
from prettytable import PrettyTable

from Scripts.Common.VLFunctions import MeshInfo


# NetPU architecture
class NetPU(nn.Module):
    def __init__(self):
        super(NetPU, self).__init__()
        self.fc1 = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, 128)
        self.fc3 = nn.Linear(128, 16)
        self.fc4 = nn.Linear(16, 2)
        self.drop = nn.Dropout(0.4)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.drop(x)
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        return x

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


def train(model, train_rxyz_norm, train_mu_sigma_norm, test_rxyz_norm, test_mu_sigma_norm,device,
          batch_size=32, epochs=1000, loss_func=nn.MSELoss(reduction='mean'),
          lr=0.001, GPU=False, check_epoch=50, show=50):
    # data loader
    train_dataset = Data.TensorDataset(train_rxyz_norm, train_mu_sigma_norm)
    train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # train on GPU
    if GPU:
        model = model.to(device)
        test_rxyz_norm, test_mu_sigma_norm = test_rxyz_norm.to(device), test_mu_sigma_norm.to(device)

    # optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # history
    hist = {'loss': [], 'loss_val': []}
    # epoch loop
    old=True
    for epoch in np.arange(epochs):
        # batch loop
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
        model.eval()
        loss_val = loss_func(model(test_rxyz_norm), test_mu_sigma_norm)
        model.train()

        # info epoch loop
        hist['loss'].append(loss.cpu().detach().numpy().tolist())
        hist['loss_val'].append(loss_val.cpu().detach().numpy().tolist())

        if epoch==0:
            oldavg = loss_val
            oldmodel=copy.deepcopy(model)
        elif epoch % check_epoch == 0:
            avg = np.mean(hist['loss_val'][-check_epoch:])
            print('epoch={:d}, loss={:.6f}, loss_val={:.6f}, avg.loss_val={:.6f}'.format(epoch, loss, loss_val,avg))
            if old and hist['loss_val'][epoch] > hist['loss_val'][epoch - check_epoch]:
                print('old model copied')
                oldmodel=copy.deepcopy(model)
                old=False
            if avg>oldavg:
                break
            oldavg=avg

    # info end
    print('Training finished; epoch = %d, loss = %f, loss_val = %f' % (epoch, loss, loss_val))

    # send model back to CPU
    if GPU:
        model = model.cpu()
        test_rxyz_norm, test_mu_sigma_norm = test_rxyz_norm.cpu(), test_mu_sigma_norm.cpu()

    # return history
    return hist, epoch

def DataPool(ResDir, MeshDir):
    sys.path.insert(0,ResDir)
    Parameters = reload(import_module('Parameters'))
    sys.path.pop(0)

    ERMESres = h5py.File("{}/PreAster/ERMES.rmed".format(ResDir), 'r')
    Watts = ERMESres["EM_Load/Watts"][:]
    JHNode =  ERMESres["EM_Load/JHNode"][:]
    ERMESres.close()

    # Calculate power
    CoilPower = np.sum(Watts)
    # Calculate uniformity
    Meshcls = MeshInfo("{}/{}.med".format(MeshDir,Parameters.Mesh))
    CoilFace = Meshcls.GroupInfo('CoilFace')
    Meshcls.Close()

    Std = np.std(JHNode[CoilFace.Nodes-1])

    Input = Parameters.CoilDisplacement+[Parameters.Rotation]
    Output = [CoilPower,Std]

    return Input+Output

def GetData(VL,MLdict):
    ResDirs = []
    for SubDir in os.listdir(VL.STUDY_DIR):
        ResDir = "{}/{}".format(VL.STUDY_DIR,SubDir)
        if not os.path.isdir(ResDir): continue

        ResDirs.append(ResDir)

    Pool = ProcessPool(nodes=6, workdir=VL.TMP_DIR)
    Data = Pool.map(DataPool,ResDirs,[VL.MESH_DIR]*len(ResDirs))

    return Data

def Optima(Start,model,dir,num):
    OldCoord = BestCoord = torch.tensor(Start, dtype=torch.float32) # Starting point
    OldObj = BestObj = model.predict(OldCoord)[num]

    ds = 0.00001
    lr=0.0005
    for it in range(200):
        grads = []
        for ipar in np.arange(4):
            FDp = OldCoord.clone()
            FDm = OldCoord.clone()
            FDp[ipar]+=ds
            FDm[ipar]-=ds

            with torch.no_grad():
                grad = (model.predict(FDp) - model.predict(FDm))/(2*ds)
            grads.append(grad[num])

        if dir=='max':
            NewCoord = OldCoord + lr*torch.tensor(grads, dtype=torch.float32)
        else:
            NewCoord = OldCoord - lr*torch.tensor(grads, dtype=torch.float32)

        if any(NewCoord < 0):
            NewCoord[NewCoord < 0] = 0
        if any(NewCoord > 1):
            NewCoord[NewCoord > 1] = 1

        with torch.no_grad():
            NewObj = model.predict(NewCoord)[num]

        # grd = (NewObj[0] - OldObj)/torch.norm(NewCoord-Coord)
        if it != 0 and it % 10 == 0:
            print(it, NewObj)
        # if grd < 0:
        #     break

        if NewObj > BestObj:
            BestObj = NewObj
            BestCoord = NewCoord

        OldCoord = NewCoord
        OldObj = NewObj

    return BestCoord

def main(VL, MLdict):
    # print(torch.get_num_threads())
    DataFile = "{}/Data.hdf5".format(VL.ML_DIR)
    if MLdict.NewData:
        Data = GetData(VL,MLdict)
        Data = np.array(Data).astype(np.float32)
        MLfile = h5py.File(DataFile,'w')
        MLfile.create_dataset('HIVE_Random',data=Data)
        MLfile.close()

    else :
        MLfile = h5py.File(DataFile,'r')
        Data = MLfile['HIVE_Random'][...].astype(np.float32)
        MLfile.close()

    # shuffle data here if needed
    # np.random.shuffle(Data)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')

    Total = int(np.ceil(Data.shape[0]*MLdict.DataPrcnt))
    NbTrain = int(np.ceil(Total*MLdict.SplitRatio))
    NbTest = Total-NbTrain

    TrainData = Data[:NbTrain,:]
    TestData = Data[NbTrain:Total,:]

    DataMin = TrainData.min(axis=0)
    DataMax = TrainData.max(axis=0)

    InputRange = np.array([DataMin[:4],DataMax[:4]])
    OutputRange = np.array([DataMin[4:],DataMax[4:]])

    # normalize data to training data range
    TrainData_norm = (TrainData - DataMin)/(DataMax - DataMin)
    TestData_norm = (TestData - DataMin)/(DataMax - DataMin)

    # input: (x, y, z, r)
    In_train = torch.from_numpy(TrainData_norm[:,:4])
    In_test  = torch.from_numpy(TestData_norm[:,:4])

    # output: (mu, sigma)
    Out_train = torch.from_numpy(TrainData_norm[:,4:])
    Out_test  = torch.from_numpy(TestData_norm[:,4:])

    ModelFile = '{}/model_{}.h5'.format(VL.ML_DIR, MLdict.Name) # File model will be saved to/loaded from
    if MLdict.NewModel:
        torch.manual_seed(123)
        # create model instance
        model = NetPU()

        # train the model
        model.train()

        history, epoch_stop = train(model, In_train, Out_train,
        		            In_test, Out_test, device,
                            batch_size=500, epochs=2000, lr=.0001, check_epoch=50,
                            GPU=False)

        model.eval()

        torch.save(model, ModelFile)

        plt.figure(dpi=100)
        plt.plot(history['loss'][0:epoch_stop], label='loss')
        plt.plot(history['loss_val'][0:epoch_stop], label='loss_val')
        plt.legend()
        plt.show()

    else:
        model = torch.load(ModelFile)

    model.eval()

    # prediction & scale to actual value
    NN_train = model.predict(In_train)*(OutputRange[1] - OutputRange[0]) + OutputRange[0]
    NN_test = model.predict(In_test)*(OutputRange[1] - OutputRange[0]) + OutputRange[0]

    # Error percentage for test & train using NN
    train_error = (100*(NN_train - TrainData[:,4:])/TrainData[:,4:]).detach().numpy()
    test_error = (100*(NN_test - TestData[:,4:])/TestData[:,4:]).detach().numpy()

    if getattr(MLdict,'ShowMetric',False):
        bins = list(range(-20,21))
        range5 = np.arange(bins.index(-5),bins.index(5))
        range10 = np.arange(bins.index(-10),bins.index(10))

        # Testing metrics
        Ptest_Hist = np.histogram(test_error[:,0],bins=bins)[0]
        Utest_Hist = np.histogram(test_error[:,1],bins=bins)[0]
        Cmbtest_Hist = np.histogram(np.abs(test_error).mean(axis=1),bins=bins)[0]
        Ptest5 = 100*Ptest_Hist[range5].sum()/NbTest
        Ptest10 = 100*Ptest_Hist[range10].sum()/NbTest
        Utest5 = 100*Utest_Hist[range5].sum()/NbTest
        Utest10 = 100*Utest_Hist[range10].sum()/NbTest
        Cmbtest5 = 100*Cmbtest_Hist[range5].sum()/NbTest
        Cmbtest10 = 100*Cmbtest_Hist[range10].sum()/NbTest

        TestTb = PrettyTable(['Output','+-5%','+-10%'],float_format=".3")
        TestTb.title = "Metric for test data"
        TestTb.add_row(["Power", Ptest5, Ptest10])
        TestTb.add_row(["Uniformity", Utest5, Utest10])
        TestTb.add_row(["Combined", Cmbtest5, Cmbtest10])
        print()
        print(TestTb)
        print()

        # Training metrics
        Ptrain_Hist = np.histogram(train_error[:,0],bins=bins)[0]
        Utrain_Hist = np.histogram(train_error[:,1],bins=bins)[0]
        Cmbtrain_Hist = np.histogram(np.abs(train_error).mean(axis=1),bins=bins)[0]
        Ptrain5 = 100*Ptrain_Hist[range5].sum()/NbTrain
        Ptrain10 = 100*Ptrain_Hist[range10].sum()/NbTrain
        Utrain5 = 100*Utrain_Hist[range5].sum()/NbTrain
        Utrain10 = 100*Utrain_Hist[range10].sum()/NbTrain
        Cmbtrain5 = 100*Cmbtrain_Hist[range5].sum()/NbTrain
        Cmbtrain10 = 100*Cmbtrain_Hist[range10].sum()/NbTrain

        TrainTb = PrettyTable(['Output','+-5%','+-10%'],float_format=".3")
        TrainTb.title = "Metric for training data"
        TrainTb.add_row(["Power", Ptrain5, Ptrain10])
        TrainTb.add_row(["Uniformity", Utrain5, Utrain10])
        TrainTb.add_row(["Combined", Cmbtrain5, Cmbtrain10])
        print()
        print(TrainTb)
        print()

        fig,ax = plt.subplots(2, 2,figsize=(10,15))
        ax[0,0].hist(train_error[:, 0],bins=bins)
        ax[0,0].set_title('Power_train')
        ax[0,0].set_xlabel('Error')
        ax[0,0].set_ylabel('Count')
        Xlim = max(np.absolute(ax[0,0].get_xlim()))
        ax[0,1].hist(test_error[:, 0],bins=bins)
        ax[0,1].set_title('Power_test')
        ax[0,1].set_xlabel('Error')
        ax[0,1].set_ylabel('Count')
        Xlim = max(max(np.absolute(ax[0,1].get_xlim())),Xlim)
        ax[1,0].hist(train_error[:, 1],bins=bins)
        ax[1,0].set_title('Uniform_train')
        ax[1,0].set_xlabel('Error')
        ax[1,0].set_ylabel('Count')
        Xlim = max(max(np.absolute(ax[1,0].get_xlim())),Xlim)
        ax[1,1].hist(test_error[:, 1],bins=bins)
        ax[1,1].set_title('Uniform_test')
        ax[1,1].set_xlabel('Error')
        ax[1,1].set_ylabel('Count')
        Xlim = max(max(np.absolute(ax[1,1].get_xlim())),Xlim)

        plt.setp(ax, xlim=(-Xlim,Xlim))
        plt.show()


    GridSeg = 2
    disc = np.linspace(0, 1, GridSeg+1)
    # Stores data in a logical mesh grid
    grid = np.meshgrid(disc, disc, disc, disc,indexing='ij')
    grid = np.moveaxis(np.array(grid),0,-1) #grid point is now the last axis
    ndim = grid.ndim - 1
    # unroll grid so it can be passed to model
    grid = grid.reshape([disc.shape[0]**ndim,ndim])

    with torch.no_grad():
        NNGrid = model.predict(torch.tensor(grid, dtype=torch.float32))
    NNGrid = NNGrid.detach().numpy()

    # # convert grid points and results back to grid format (if desired)
    # # Could be useful for querying
    # grid = grid.reshape([disc.shape[0]]*ndim+[grid.shape[1]])
    # NNGrid = NNGrid.detach().numpy()
    # NNGrid = NNGrid.reshape([disc.shape[0]]*ndim+[NNGrid.shape[1]])

    # get location of min & max for outputs
    MaxPower_Ix, MaxVar_Ix  = np.argmax(NNGrid,axis=0)
    MinPower_Ix, MinVar_Ix  = np.argmin(NNGrid,axis=0)

    # Get the value for power & variation at max. power & variation points
    _Power, Max_Var = (NNGrid[MaxVar_Ix,:]*(OutputRange[1]-OutputRange[0]) + OutputRange[0])

    PMaxGridCd = grid[MaxPower_Ix,:]
    # Start at this point and find optima
    PMaxCd = Optima(PMaxGridCd, model, 'max', 0)
    PMax, VarPMax = (model.predict(PMaxCd)*(OutputRange[1] - OutputRange[0]) + OutputRange[0]).detach().numpy()
    PMin = NNGrid[MinPower_Ix,:]
    print("Maximum power available is {:.3f} watts".format(PMax))

    # VMaxGridCd = grid[MaxVar_Ix,:]
    # VMaxCd = Optima(VMaxGridCd, model, 'max', 1)
    # VMax = model.predict(VMaxCd)[1]
    # VMin = NNGrid[MinVar_Ix,:]

    ds = 0.001
    lr=0.0005

    alpha=1
    mu=0.5

    DesPower = 500
    P_norm = ((DesPower - OutputRange[0])/(OutputRange[1]-OutputRange[0]))[0]

    print("Targeting {} watts of power".format(DesPower))
    print(P_norm)

    def Grad1(model, input):
        input = input.detach().numpy()
        for i in range(1,5):
            fc = getattr(model,'fc{}'.format(i))
            w = fc.weight.detach().numpy()
            b = fc.bias.detach().numpy()
            out = w.dot(input)+b
            diag = np.copy(out)
            diag[diag>=0] = 1
            diag[diag<0] = 0.01
            gd = diag[:,None]*w
            if i==1:
                cumul = gd
            else :
                cumul = gd.dot(cumul)

            out[out<0] = out[out<0]*0.01
            input = out

        return cumul

    def GradLagr(model, x, mu, P_norm, alpha):
        Sgrad = Grad1(model, x)
        Lgrads = -(alpha*Sgrad[0,:]-(1-alpha)*Sgrad[1,:]) + mu*Sgrad[0,:]
        mugrad = model.predict(x)[0]-P_norm
        return np.append(Lgrads,mugrad.detach().numpy())


    Mag = np.linalg.norm
    # Ix = np.argmin(np.abs(NNGrid[:,0]-P_norm))
    # OldCoord = torch.tensor(grid[Ix,:],dtype=torch.float32)
    OldCoord = torch.tensor([0.5,0.5,0.5,0.5], dtype=torch.float32)
    Output = model.predict(OldCoord)
    Score = alpha*Output[0] + (1-alpha)*(1-Output[1])
    print(Output, Score)

    for i in range(3200):
        G_grads = []
        for ipar in np.arange(4):
            FDp = OldCoord.clone()
            FDm = OldCoord.clone()
            FDp[ipar]+=ds
            FDm[ipar]-=ds

            G_grad = (Mag(GradLagr(model,FDp,mu,P_norm,alpha)) - Mag(GradLagr(model,FDm,mu,P_norm,alpha)))/(2*ds)
            G_grads.append(G_grad)

        mp = (Mag(GradLagr(model,FDp,mu+ds,P_norm,alpha)) - Mag(GradLagr(model,FDp,mu-ds,P_norm,alpha)))/(2*ds)

        OldCoord = OldCoord - lr*torch.tensor(G_grads, dtype=torch.float32)
        mu = mu - lr*mp

        if i % 50 == 0 and i!=0:
            print(i, Mag(GradLagr(model,OldCoord,mu,P_norm,alpha)), model.predict(OldCoord).detach().numpy())
            print(model.predict(OldCoord)[0]>P_norm)


    # L_grads = []
    # for ipar in np.arange(4):
    #     FDp = OldCoord.clone()
    #     FDm = OldCoord.clone()
    #     FDp[ipar]+=ds
    #     FDm[ipar]-=ds
    #     with torch.no_grad():
    #         grad = (model.predict(FDp) - model.predict(FDm))/(2*ds)
    #     L_grads.append(grad.detach().numpy())
    #
    # print(np.array(L_grads))
    # gd = 1
    # input = np.array([0.5,0.5,0.5,0.5])
    #
    #
    # print(cumul.T)


    # test = w1*np.array([0.5,0.5,0.5,0.5]) + b1[:,None]
    # print(test)

    # def grads(model, x, mu, P_norm):
    #     ds = 0.00001
    #     L_grads = []
    #     for ipar in np.arange(4):
    #         FDp = x.clone()
    #         FDm = x.clone()
    #         FDp[ipar]+=ds
    #         FDm[ipar]-=ds
    #         with torch.no_grad():
    #             grad = (model.predict(FDp) - model.predict(FDm))/(2*ds)
    #
    #         L_grad = -(alpha*grad[0]-(1-alpha)*grad[1]) + mu*grad[0]
    #         L_grads.append(L_grad)
    #
    #     L_grads.append(model.predict(x)[0]-P_norm)
    #
    #     return L_grads
    #
    # print(np.linalg.norm(grads(model,OldCoord,mu,P_norm)))
    #


    # MaxGridCd = MaxGridCd*(InputRange[1]-InputRange[0]) + InputRange[0]
    # MaxGridPower, Variation = GridOut_norm[MaxPower_Ix,:]*(OutputRange[1]-OutputRange[0]) + OutputRange[0]
    # print("Best power based on grid point")
    # print(MaxGridCd, MaxGridPower, Variation)
    #
    # MaxCd = (BestCoord*(InputRange[1]-InputRange[0]) + InputRange[0]).detach().numpy()
    # MaxPower, Variation = (model.predict(BestCoord, device)*(OutputRange[1]-OutputRange[0]) + OutputRange[0]).detach().numpy()
    # print("Best power after gradient ascent")
    # print(MaxCd, MaxPower, Variation)


    '''

    # MaxCd = MaxGridCd
    # BestCoord = torch.tensor(xyzr[MaxPower_Ix,:], dtype=torch.float32)
    from types import SimpleNamespace as Namespace

    ParaDict = {'CoilType':'HIVE',
                'CoilDisplacement':MaxCd[:3].tolist(),
                'Rotation':MaxCd[3],
                'Current':1000,
                'Frequency':1e4,
                'Materials':{'Block':'Copper_NL', 'Pipe':'Copper_NL', 'Tile':'Tungsten_NL'}}

    VL.WriteModule("{}/Parameters.py".format(VL.tmpML_DIR), ParaDict)

    Parameters = Namespace(**ParaDict)

    tstDict = {'TMP_CALC_DIR':VL.tmpML_DIR, 'MeshFile':"{}/AMAZEsample.med".format(VL.MESH_DIR),'LogFile':None,'Parameters':Parameters}
    ERMESfile = '{}/maxERMES.rmed'.format(VL.ML_DIR)
    from PreAster import devPreHIVE
    Watts, WattsPV, Elements, JHNode = devPreHIVE.SetupERMES(VL, tstDict,ERMESfile)

    # ERMESres = h5py.File(ERMESfile, 'r')
    # attrs =  ERMESres["EM_Load"].attrs
    # Elements = ERMESres["EM_Load/Elements"][:]
    #
    # Scale = (Parameters.Current/attrs['Current'])**2
    # Watts = ERMESres["EM_Load/Watts"][:]*Scale
    # WattsPV = ERMESres["EM_Load/WattsPV"][:]*Scale
    # JHNode =  ERMESres["EM_Load/JHNode"][:]*Scale
    # ERMESres.close()

    Meshcls = MeshInfo(tstDict['MeshFile'])
    CoilFace = Meshcls.GroupInfo('CoilFace')
    Meshcls.Close()
    Std = np.std(JHNode[CoilFace.Nodes-1])

    Pred = (model.predict(BestCoord, device)*(OutputRange[1]-OutputRange[0]) + OutputRange[0]).detach().numpy()
    Act = np.array([Watts.sum(),Std])
    print(Pred, Act)
    err = 100*(Pred - Act)/Act
    print(err)

    val = torch.argmax(train_mu_sigma, axis=0)
    # point = train_rxyz[val[0],:]
    # model.predict_denorm()
    point_norm = train_rxyz_norm[val[0],:]
    pred = (model.predict(point_norm,device)*(OutputRange[1]-OutputRange[0]) + OutputRange[0]).detach().numpy()
    act = train_mu_sigma[val[0],:].detach().numpy()
    err = 100*(pred - act)/act
    print(err)
    '''

    '''
    # gradient ascent descent method which doesnt seem to work
    for i in range(50000):
        L_grads = []
        for ipar in np.arange(4):
            FDp = OldCoord.clone()
            FDm = OldCoord.clone()
            FDp[ipar]+=ds
            FDm[ipar]-=ds

            with torch.no_grad():
                grad = (model.predict(FDp) - model.predict(FDm))/(2*ds)

            L_grad = -(alpha*grad[0]-(1-alpha)*grad[1]) + mu*grad[0]
            L_grads.append(L_grad)

        NewCoord = OldCoord - lr*torch.tensor(L_grads, dtype=torch.float32)
        # if any(NewCoord < 0):
        #     NewCoord[NewCoord < 0] = 0
        # if any(NewCoord > 1):
        #     NewCoord[NewCoord > 1] = 1
        mu = mu + (lr/10)*(Output[0] - P_norm)

        Output = model.predict(NewCoord)
        Score = alpha*Output[0] + (1-alpha)*(1-Output[1])
        if i % 1000 == 0 and i!=0:
            # _sc = (-(alpha*Output[0] + (1-alpha)*(1-Output[1])) + mu*(Output[0]-P_norm)).detach().numpy()
            print(i, Score, Output.detach().numpy(), mu)

        OldCoord = NewCoord

    print(Output.detach().numpy())
    '''
