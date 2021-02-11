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

    def GradNN(self, input):
        '''
        Function which returns the NN gradient at N input points
        Input: 2-d array of points with dimension (N ,NbInput)
        Output: 3-d array of partial derivatives (NbOutput, NbInput, N)
        '''
        input = np.atleast_2d(input)
        for i in range(1,5):
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

    # loss tensor
    IndLoss = nn.MSELoss(reduction='none')
    # history
    hist = {'loss_batch': [], 'loss_train': [], 'loss_val': []}
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
        loss_train = loss_func(model(train_rxyz_norm), train_mu_sigma_norm)
        model.train()

        # # loss value for power and uniformity seperately
        # loss_val_sep = torch.mean(IndLoss(model(test_rxyz_norm), test_mu_sigma_norm),dim=0)
        # loss_train_sep = torch.mean(IndLoss(model(train_rxyz_norm), train_mu_sigma_norm),dim=0)

        # info epoch loop
        hist['loss_batch'].append(loss.cpu().detach().numpy().tolist())
        hist['loss_val'].append(loss_val.cpu().detach().numpy().tolist())
        hist['loss_train'].append(loss_train.cpu().detach().numpy().tolist())

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
            # if avg>oldavg:
            #     break
            oldavg=avg

    # info end
    print('Training finished; epoch = %d, loss = %f, loss_val = %f' % (epoch, loss, loss_val))

    # send model back to CPU
    if GPU:
        model = model.cpu()
        test_rxyz_norm, test_mu_sigma_norm = test_rxyz_norm.cpu(), test_mu_sigma_norm.cpu()

    # return history
    return hist, epoch


def DataPool(ResDir, MeshDir,):
    # function which is used by ProcessPool map
    sys.path.insert(0,ResDir)
    Parameters = reload(import_module('Parameters'))
    sys.path.pop(0)

    ERMESres = h5py.File("{}/PreAster/ERMES.rmed".format(ResDir), 'r')
    Watts = ERMESres["EM_Load/Watts"][:]
    JHNode =  ERMESres["EM_Load/JHNode"][:]
    ERMESres.close()
    print(ResDir)
    # Calculate power
    CoilPower = np.sum(Watts)

    # # method 1 looks at the stdev
    # Meshcls = MeshInfo("{}/{}.med".format(MeshDir,Parameters.Mesh))
    # CoilFace = Meshcls.GroupInfo('CoilFace')
    # Meshcls.Close()
    # Uniformity = np.std(JHNode[CoilFace.Nodes-1])

    # method 2
    # Meshcls = MeshInfo("{}/{}.med".format(MeshDir,Parameters.Mesh))
    # CoilFace = Meshcls.GroupInfo('CoilFace')
    # Meshcls.Close()
    # Issue here is that it scales each sample individually
    # Scale JH to 0 to 2 so that StDev must be in range [0,1]
    # JHMax, JHMin = JHNode.max(), JHNode.min()
    # JHNode = 2*(JHNode - JHMin)/(JHMax-JHMin)
    # # flips StDev so that 1 is best score and 0 is the worst
    # Uniformity = 1 - np.std(JHNode[CoilFace.Nodes-1])
    tst = np.inf
    JHNode /= Parameters.Current **2
    Meshcls = MeshInfo("{}/{}.med".format(MeshDir,Parameters.Mesh))
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

    Input = Parameters.CoilDisplacement+[Parameters.Rotation]
    Output = [CoilPower,Uniformity]

    return Input+Output

def GetData(VL, ML):
    ResDirs = []
    DataDir = VL.STUDY_DIR
    for SubDir in natsorted(os.listdir(DataDir)):
        ResDir = "{}/{}".format(DataDir,SubDir)
        if not os.path.isdir(ResDir): continue

        ResDirs.append(ResDir)

    Pool = ProcessPool(nodes=10, workdir=VL.TEMP_DIR)
    Data = Pool.map(DataPool,ResDirs,[VL.MESH_DIR]*len(ResDirs))

    return Data

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


def func(X, model, alpha):
    '''
    This is the function which we look to optimise, which is a weighted geometric
    average of power & uniformity
    '''
    X = torch.tensor(X, dtype=torch.float32)
    PV = model.predict(X).detach().numpy()
    score = alpha*PV[0] + (1-alpha)*(PV[1])
    return -(score)

def dfunc(X, model, alpha):
    '''
    Derivative of 'func' w.r.t. the inputs x,y,z,r
    '''
    # Grad = Gradold(model, X)
    Grad = model.GradNN(X)[0]
    dscore = alpha*Grad[0,:] + (1-alpha)*Grad[1,:]
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

def main(VL, MLdict):
    ML = MLdict["Parameters"]
    # print(torch.get_num_threads())

    # File where all data is stored
    DataFile = "{}/Data.hdf5".format(VL.ML_DIR)

    # Create new data
    if ML.CreateData:
        Data = GetData(VL, ML)
        Data = np.array(Data)
        MLfile = h5py.File(DataFile,'a')
        if VL.StudyName not in MLfile.keys():
            StudyGrp = MLfile.create_group(VL.StudyName)
        else :
            StudyGrp = MLfile[VL.StudyName]
        if ML.DataName in StudyGrp.keys():
            del StudyGrp[ML.DataName]
        StudyGrp.create_dataset(ML.DataName, data=Data)
        MLfile.close()

    # Get Train and Test Data
    DataPrcnt = getattr(ML,'DataPrcnt',1)
    DataSplit = getattr(ML,'DataSplit',0.7)

    MLData = h5py.File(DataFile,'r')

    if ML.TrainData in MLData:
        TrainData = MLData[ML.TrainData][:]
    elif "{}/{}".format(VL.StudyName,ML.TrainData) in MLData:
        TrainData = MLData["{}/{}".format(VL.StudyName,ML.TrainData)][:]
    else :
        sys.exit("Data not found")
    TrainNb = int(np.ceil(TrainData.shape[0]*DataPrcnt))

    if hasattr(ML,'TestData'):
        # Get data from elsewhere
        TestNb = int(np.ceil(TrainNb*(1-DataSplit)/DataSplit))
        TestData = MLData[ML.TestData][:]
        TestData = TestData[:TestNb,:]
    else :
        TestNb = int(np.ceil(TrainNb*(1-DataSplit)))
        TrainNb -= TestNb
        TestData = TrainData[TrainNb:,:]
        TrainData = TrainData[:TrainNb,:]

    MLData.close()

    # shuffle data here if needed
    # np.random.shuffle(Data)

    # Convert data to float32 for PyTorch
    TrainData = TrainData.astype('float32')
    TestData = TestData.astype('float32')

    # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')

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

    ModelFile = '{}/model.h5'.format(MLdict["CALC_DIR"]) # File model will be saved to/loaded from
    if ML.Train:
        torch.manual_seed(123)
        # create model instance
        model = NetPU()

        # train the model
        model.train()

        history, epoch_stop = train(model, In_train, Out_train,
        		            In_test, Out_test, device,
                            batch_size=ML.BatchSize, epochs=ML.NbEpoch, lr=.0001, check_epoch=50,
                            GPU=False)

        # model.eval()

        torch.save(model, ModelFile)

        plt.figure(dpi=100)
        plt.plot(history['loss_train'][0:epoch_stop], label='loss_train')
        plt.plot(history['loss_val'][0:epoch_stop], label='loss_val')
        plt.legend()
        plt.savefig("{}/ModelConvergence.png".format(MLdict["CALC_DIR"]))
        plt.close()

    else:
        model = torch.load(ModelFile)

    model.eval()


    from adaptive import LearnerND
    import adaptive.learner.learnerND as LND

    def fn(xyzr):
    	xyzr = torch.tensor(xyzr, dtype=torch.float32)
    	out = model.predict(xyzr).detach().numpy()
    	return out*(OutputRange[1] - OutputRange[0]) + OutputRange[0]
    # test comment
    Total = 2000
    batch = 50
    bins = 11
    LossMetric = 'triangle' # uniform, value, triangle

    if LossMetric == 'uniform': LossMet = LND.uniform_loss
    elif LossMetric == 'value': LossMet = LND.default_loss
    elif LossMetric == 'triangle': LossMet = LND.triangle_loss

    learner = LearnerND(lambda l:1, bounds=[(0, 1),(0, 1),(0,1),(0,1)], loss_per_simplex=LossMet)

    while learner.npoints<Total:
        print(learner.npoints)
        Xs,ls = learner.ask(batch)
        for x in Xs:
            y = fn(x)[0]
            learner.tell(x,y)

    fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True, sharey=True, dpi=200, figsize=(4,10))
    arr = np.array(list(learner.data.keys()))

    for i in range(4):
    	ax[i].hist(arr[:,i], color = "skyblue", bins=bins)
    # plt.show()

    Slice = 'xy'
    Res = 'Power' # 'Uniformity'
    Component = 'va'
    MajorN = 7
    MinorN = 20

    InTag, OutTag = ['x','y','z','r'],['power','uniformity']
    imslice = [InTag.index(ax.lower()) for ax in Slice]
    ResAx = OutTag.index(Res.lower())

    DiscSl = np.linspace(0+0.5*1/MinorN,1-0.5*1/MinorN,MinorN)
    DiscAx = np.linspace(0,1,MajorN)
    # DiscAx = np.linspace(0+0.5*1/MajorN,1-0.5*1/MajorN,MajorN)

    disc = [DiscAx]*4
    disc[imslice[0]] = disc[imslice[1]] = DiscSl
    grid = np.meshgrid(*disc, indexing='ij')
    grid = np.moveaxis(np.array(grid),0,-1) #grid point is now the last axis
    ndim = grid.ndim - 1

    # unroll grid so it can be passed to model
    _disc = [dsc.shape[0] for dsc in disc]
    grid = grid.reshape([np.prod(_disc),ndim])

    # decide what component of Res to print - gradient (magnitude or individual) or the value
    if Component.lower() == 'grad':
        imres = model.GradNN(grid)
        imres = np.linalg.norm(imres, axis=2)
    elif Component.lower().startswith('grad'):
        ax = InTag.index(Component[-1].lower())
        imres = model.GradNN(grid)
        imres = imres[:,:,ax]
    else :
        with torch.no_grad():
            imres = model.predict(torch.tensor(grid, dtype=torch.float32)).detach().numpy()
            imres = imres*(OutputRange[1] - OutputRange[0]) + OutputRange[0]

    IMMin, IMMax = imres.min(axis=0), imres.max(axis=0)

    grid = grid.reshape(_disc+[grid.shape[1]])
    grid = grid*(InputRange[1]-InputRange[0]) + InputRange[0]

    imres = imres.reshape(_disc+[*imres.shape[1:]])

    _ix = [i for i in range(len(InTag)) if i not in imslice]

    _sl = [0]*len(InTag) + [_ix[0]]
    _sl[_ix[0]] = slice(None)
    arr1 = grid[tuple(_sl)]
    # arr1 = arr1[:1]
    _sl = [0]*len(InTag) + [_ix[1]]
    _sl[_ix[1]] = slice(None)
    arr2 = grid[tuple(_sl)]

    fig, ax = plt.subplots(nrows=arr1.size, ncols=arr2.size, sharex=True, sharey=True, dpi=200, figsize=(4,6))
    ax = np.atleast_2d(ax)

    fig.subplots_adjust(right=0.8)
    for it1,nb1 in enumerate(arr1):
        ax[-(it1+1), 0].set_ylabel('{:.4f}'.format(nb1), fontsize=6)
        for it2,nb2 in enumerate(arr2):
            sl = [slice(None)]*len(InTag) + [ResAx]
            sl[_ix[0]],sl[_ix[1]]  = it1, it2
            tst = imres[tuple(sl)].T
            m1,m2 = tst.max(),tst.min()

            Im = ax[-(it1+1),it2].imshow(imres[tuple(sl)].T, cmap = 'coolwarm', vmin=IMMin[ResAx], vmax=IMMax[ResAx], origin='lower')
            ax[-(it1+1),it2].set_xticks([])
            ax[-(it1+1),it2].set_yticks([])

            ax[-1, it2].set_xlabel("{} {:.4f}".format(InTag[_ix[1]],nb2), fontsize=6)

    if 1:
        batch = 90000
        for j in range(arr.shape[0] // batch + 1):
            _batch = arr[:(j+1)*batch,:]*(InputRange[1]-InputRange[0]) + InputRange[0]

            for it1,nb1 in enumerate(arr1):
                if it1 == 0: bl1 = _batch[:,_ix[0]] <= np.mean(arr1[:2])
                elif it1 == arr1.shape[0]-1: bl1 = _batch[:,_ix[0]] > np.mean(arr1[-2:])
                else: bl1 = (np.mean(arr1[it1-1:it1+1]) < _batch[:,_ix[0]])*(_batch[:,_ix[0]] <= np.mean(arr1[it1:it1+2]))
                for it2,nb2 in enumerate(arr2):
                    if it2 == 0: bl2 = _batch[:,_ix[1]] <= np.mean(arr2[:2])
                    elif it2 == arr2.shape[0]-1: bl2 = _batch[:,_ix[1]] > np.mean(arr2[-2:])
                    else: bl2 = (np.mean(arr2[it2-1:it2+1]) < _batch[:,_ix[1]])*(_batch[:,_ix[1]] <= np.mean(arr2[it2:it2+2]))
                    bl = bl1*bl2
                    dat = _batch[bl,:]

                    sl1 = (dat[:,imslice[0]] - InputRange[0,imslice[0]])/(InputRange[1,imslice[0]] - InputRange[0,imslice[0]])
                    min1, max1 = ax[it1,it2].get_xlim()
                    sl1 = min1 + sl1*(max1 - min1)

                    sl2 = (dat[:,imslice[1]] - InputRange[0,imslice[1]])/(InputRange[1,imslice[1]] - InputRange[0,imslice[1]])
                    min2, max2 = ax[it1,it2].get_ylim()
                    sl2 = min2 + sl2*(max2 - min2)

                    Sc = ax[-(it1+1),it2].scatter(sl1,sl2, c='r', marker = 'x', s=2)
            plt.pause(0.1)
        cbar_ax = fig.add_axes([0.85, 0.05, 0.05, 0.4])
        fig.colorbar(Sc, cax=cbar_ax)

    cbar_ax = fig.add_axes([0.85, 0.55, 0.05, 0.4])
    bnds = np.linspace(*Im.get_clim(),12)
    ticks = np.linspace(*Im.get_clim(),6)
    fig.colorbar(Im, boundaries=bnds, ticks=ticks, cax=cbar_ax)

    fig.suptitle(Res.capitalize())
    fig.text(0.5, 0.04, Slice[0].capitalize(), ha='center')
    fig.text(0.04, 0.5, Slice[1].capitalize(), va='center', rotation='vertical')

    plt.show()

    sys.exit()

    # prediction & scale to actual value
    NN_train = model.predict(In_train)*(OutputRange[1] - OutputRange[0]) + OutputRange[0]
    NN_test = model.predict(In_test)*(OutputRange[1] - OutputRange[0]) + OutputRange[0]

    # Error percentage for test & train using NN
    train_error = (NN_train - TrainData[:,4:]).detach().numpy()
    test_error = (NN_test - TestData[:,4:]).detach().numpy()

    train_error_perc = 100*train_error/TrainData[:,4:]
    test_error_perc = 100*test_error/TestData[:,4:]

    if getattr(ML,'ShowMetric',False):
        bins = list(range(-20,21))
        range5 = np.arange(bins.index(-5),bins.index(5))
        range10 = np.arange(bins.index(-10),bins.index(10))

        # Testing metrics
        Ptest_Hist = np.histogram(test_error_perc[:,0],bins=bins)[0]
        Utest_Hist = np.histogram(test_error_perc[:,1],bins=bins)[0]
        Cmbtest_Hist = np.histogram(np.abs(test_error_perc).mean(axis=1),bins=bins)[0]
        Ptest5 = 100*Ptest_Hist[range5].sum()/TestNb
        Ptest10 = 100*Ptest_Hist[range10].sum()/TestNb
        Utest5 = 100*Utest_Hist[range5].sum()/TestNb
        Utest10 = 100*Utest_Hist[range10].sum()/TestNb
        Cmbtest5 = 100*Cmbtest_Hist[range5].sum()/TestNb
        Cmbtest10 = 100*Cmbtest_Hist[range10].sum()/TestNb

        TestTb = PrettyTable(['Output','+-5%','+-10%'],float_format=".3")
        TestTb.title = "Metric for test data"
        TestTb.add_row(["Power", Ptest5, Ptest10])
        TestTb.add_row(["Uniformity", Utest5, Utest10])
        TestTb.add_row(["Combined", Cmbtest5, Cmbtest10])
        print()
        print(TestTb)
        print()

        # Training metrics
        Ptrain_Hist = np.histogram(train_error_perc[:,0],bins=bins)[0]
        Utrain_Hist = np.histogram(train_error_perc[:,1],bins=bins)[0]
        Cmbtrain_Hist = np.histogram(np.abs(train_error_perc).mean(axis=1),bins=bins)[0]
        Ptrain5 = 100*Ptrain_Hist[range5].sum()/TrainNb
        Ptrain10 = 100*Ptrain_Hist[range10].sum()/TrainNb
        Utrain5 = 100*Utrain_Hist[range5].sum()/TrainNb
        Utrain10 = 100*Utrain_Hist[range10].sum()/TrainNb
        Cmbtrain5 = 100*Cmbtrain_Hist[range5].sum()/TrainNb
        Cmbtrain10 = 100*Cmbtrain_Hist[range10].sum()/TrainNb

        TrainTb = PrettyTable(['Output','+-5%','+-10%'],float_format=".3")
        TrainTb.title = "Metric for training data"
        TrainTb.add_row(["Power", Ptrain5, Ptrain10])
        TrainTb.add_row(["Uniformity", Utrain5, Utrain10])
        TrainTb.add_row(["Combined", Cmbtrain5, Cmbtrain10])
        print()
        print(TrainTb)
        print()

        fig,ax = plt.subplots(2, 2,figsize=(10,15))
        ax[0,0].hist(train_error_perc[:, 0],bins=bins)
        ax[0,0].set_title('Power_train')
        ax[0,0].set_xlabel('Error')
        ax[0,0].set_ylabel('Count')
        Xlim = max(np.absolute(ax[0,0].get_xlim()))
        ax[0,1].hist(test_error_perc[:, 0],bins=bins)
        ax[0,1].set_title('Power_test')
        ax[0,1].set_xlabel('Error')
        ax[0,1].set_ylabel('Count')
        Xlim = max(max(np.absolute(ax[0,1].get_xlim())),Xlim)
        ax[1,0].hist(train_error_perc[:, 1],bins=bins)
        ax[1,0].set_title('Uniform_train')
        ax[1,0].set_xlabel('Error')
        ax[1,0].set_ylabel('Count')
        Xlim = max(max(np.absolute(ax[1,0].get_xlim())),Xlim)
        ax[1,1].hist(test_error_perc[:, 1],bins=bins)
        ax[1,1].set_title('Uniform_test')
        ax[1,1].set_xlabel('Error')
        ax[1,1].set_ylabel('Count')
        Xlim = max(max(np.absolute(ax[1,1].get_xlim())),Xlim)

        plt.setp(ax, xlim=(-Xlim,Xlim))
        plt.show()
        bins = 20
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

        # plt.setp(ax, xlim=(-Xlim,Xlim))
        plt.show()


        Slice = 'xy'
        Res = 'Power' # 'Uniformity'
        Component = 'va'
        MajorN = 7
        MinorN = 20

        InTag, OutTag = ['x','y','z','r'],['power','uniformity']
        imslice = [InTag.index(ax.lower()) for ax in Slice]
        ResAx = OutTag.index(Res.lower())

        DiscSl = np.linspace(0+0.5*1/MinorN,1-0.5*1/MinorN,MinorN)
        DiscAx = np.linspace(0,1,MajorN)
        # DiscAx = np.linspace(0+0.5*1/MajorN,1-0.5*1/MajorN,MajorN)

        disc = [DiscAx]*4
        disc[imslice[0]] = disc[imslice[1]] = DiscSl
        grid = np.meshgrid(*disc, indexing='ij')
        grid = np.moveaxis(np.array(grid),0,-1) #grid point is now the last axis
        ndim = grid.ndim - 1

        # unroll grid so it can be passed to model
        _disc = [dsc.shape[0] for dsc in disc]
        grid = grid.reshape([np.prod(_disc),ndim])

        # decide what component of Res to print - gradient (magnitude or individual) or the value
        if Component.lower() == 'grad':
            imres = model.GradNN(grid)
            imres = np.linalg.norm(imres, axis=2)
        elif Component.lower().startswith('grad'):
            ax = InTag.index(Component[-1].lower())
            imres = model.GradNN(grid)
            imres = imres[:,:,ax]
        else :
            with torch.no_grad():
                imres = model.predict(torch.tensor(grid, dtype=torch.float32)).detach().numpy()
                imres = imres*(OutputRange[1] - OutputRange[0]) + OutputRange[0]

        IMMin, IMMax = imres.min(axis=0), imres.max(axis=0)

        grid = grid.reshape(_disc+[grid.shape[1]])
        grid = grid*(InputRange[1]-InputRange[0]) + InputRange[0]

        imres = imres.reshape(_disc+[*imres.shape[1:]])

        _ix = [i for i in range(len(InTag)) if i not in imslice]

        _sl = [0]*len(InTag) + [_ix[0]]
        _sl[_ix[0]] = slice(None)
        arr1 = grid[tuple(_sl)]
        # arr1 = arr1[:1]
        _sl = [0]*len(InTag) + [_ix[1]]
        _sl[_ix[1]] = slice(None)
        arr2 = grid[tuple(_sl)]

        fig, ax = plt.subplots(nrows=arr1.size, ncols=arr2.size, sharex=True, sharey=True, dpi=200, figsize=(4,6))
        ax = np.atleast_2d(ax)

        fig.subplots_adjust(right=0.8)
        for it1,nb1 in enumerate(arr1):
            ax[-(it1+1), 0].set_ylabel('{:.4f}'.format(nb1), fontsize=6)
            for it2,nb2 in enumerate(arr2):
                sl = [slice(None)]*len(InTag) + [ResAx]
                sl[_ix[0]],sl[_ix[1]]  = it1, it2
                tst = imres[tuple(sl)].T
                m1,m2 = tst.max(),tst.min()
                print(m1,m2,m1-m2)

                Im = ax[-(it1+1),it2].imshow(imres[tuple(sl)].T, cmap = 'coolwarm', vmin=IMMin[ResAx], vmax=IMMax[ResAx], origin='lower')
                ax[-(it1+1),it2].set_xticks([])
                ax[-(it1+1),it2].set_yticks([])

                ax[-1, it2].set_xlabel("{} {:.4f}".format(InTag[_ix[1]],nb2), fontsize=6)

        if 0:
            errmin, errmax = train_error.min(axis=0)[ResAx], train_error.max(axis=0)[ResAx]
            errbnd = max(abs(errmin),abs(errmax))
            batch = 90000
            for j in range(TrainData.shape[0] // batch + 1):
                _batch = TrainData[:(j+1)*batch,:]
                for it1,nb1 in enumerate(arr1):
                    if it1 == 0: bl1 = _batch[:,_ix[0]] <= np.mean(arr1[:2])
                    elif it1 == arr1.shape[0]-1: bl1 = _batch[:,_ix[0]] > np.mean(arr1[-2:])
                    else: bl1 = (np.mean(arr1[it1-1:it1+1]) < _batch[:,_ix[0]])*(_batch[:,_ix[0]] <= np.mean(arr1[it1:it1+2]))
                    for it2,nb2 in enumerate(arr2):
                        if it2 == 0: bl2 = _batch[:,_ix[1]] <= np.mean(arr2[:2])
                        elif it2 == arr2.shape[0]-1: bl2 = _batch[:,_ix[1]] > np.mean(arr2[-2:])
                        else: bl2 = (np.mean(arr2[it2-1:it2+1]) < _batch[:,_ix[1]])*(_batch[:,_ix[1]] <= np.mean(arr2[it2:it2+2]))
                        bl = bl1*bl2
                        dat = _batch[bl,:]

                        sl1 = (dat[:,imslice[0]] - InputRange[0,imslice[0]])/(InputRange[1,imslice[0]] - InputRange[0,imslice[0]])
                        min1, max1 = ax[it1,it2].get_xlim()
                        sl1 = min1 + sl1*(max1 - min1)

                        sl2 = (dat[:,imslice[1]] - InputRange[0,imslice[1]])/(InputRange[1,imslice[1]] - InputRange[0,imslice[1]])
                        min2, max2 = ax[it1,it2].get_ylim()
                        sl2 = min2 + sl2*(max2 - min2)

                        cl = train_error[:(j+1)*batch,ResAx][bl]
                        cmap = cm.get_cmap('PiYG')

                        Sc = ax[-(it1+1),it2].scatter(sl1,sl2, c=cl, marker = 'x', cmap=cmap, vmin = -errbnd, vmax = errbnd, s=2)
                plt.pause(0.1)
            cbar_ax = fig.add_axes([0.85, 0.05, 0.05, 0.4])
            fig.colorbar(Sc, cax=cbar_ax)

        cbar_ax = fig.add_axes([0.85, 0.55, 0.05, 0.4])
        bnds = np.linspace(*Im.get_clim(),12)
        ticks = np.linspace(*Im.get_clim(),6)
        fig.colorbar(Im, boundaries=bnds, ticks=ticks, cax=cbar_ax)

        fig.suptitle(Res.capitalize())
        fig.text(0.5, 0.04, Slice[0].capitalize(), ha='center')
        fig.text(0.04, 0.5, Slice[1].capitalize(), va='center', rotation='vertical')

        plt.show()

    # In = TrainData[:,:4]

    # print(grid[0,0,0,0],grads[0,0,0,0])
    # print(grid[0,1,0,0],grads[0,1,0,0])
    # print(grid[0,0,0,1],grads[0,0,0,1])
    # with torch.no_grad():
    #     NNGrid = model.predict(torch.tensor(grid, dtype=torch.float32))
    # NNGrid = NNGrid.detach().numpy()
    # print(NNGrid)


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

    if False:
        # convert grid points and results back to grid format (if desired)
        # Could be useful for querying
        grid = grid.reshape([disc.shape[0]]*ndim+[grid.shape[1]])
        NNGrid = NNGrid.detach().numpy()
        NNGrid = NNGrid.reshape([disc.shape[0]]*ndim+[NNGrid.shape[1]])

    # get location of min & max from grid
    MaxPower_Ix, MaxVar_Ix  = np.argmax(NNGrid,axis=0)
    MinPower_Ix, MinVar_Ix  = np.argmin(NNGrid,axis=0)


    # optimisation with constraints
    b = (0.0,1.0)
    bnds = (b, b, b, b)

    # Get min max output range from NN. requires starting point
    X0 = [0.1,0.5,0.5,0.5]
    MaxP = minimize(MinMax, X0, args=(model,1,0), jac=dMinMax, method='SLSQP', bounds=bnds)
    MaxV = minimize(MinMax, X0, args=(model,1,1), jac=dMinMax, method='SLSQP', bounds=bnds)
    MinP = minimize(MinMax, X0, args=(model,-1,0), jac=dMinMax, method='SLSQP', bounds=bnds)
    MinV = minimize(MinMax, X0, args=(model,-1,1), jac=dMinMax, method='SLSQP', bounds=bnds)
    NNRange = np.array([[MinP.fun,MinV.fun],[-MaxP.fun,-MaxV.fun]])
    # print('Global Max/Min')
    # print(NNRange)

    NNExtrema = NNRange*(OutputRange[1] - OutputRange[0]) + OutputRange[0]
    # print(NNExtrema)


    # SHGO find all local minimums and global min
    MaxP = shgo(MinMax, bnds, args=(model,1,0), n=100, iters=1, sampling_method='sobol',minimizer_kwargs={'method':'SLSQP','jac':dMinMax,'args':(model,1,0)})
    MaxV = shgo(MinMax, bnds, args=(model,1,1), n=100, iters=1, sampling_method='sobol',minimizer_kwargs={'method':'SLSQP','jac':dMinMax,'args':(model,1,1)})
    MinP = shgo(MinMax, bnds, args=(model,-1,0), n=100, iters=1, sampling_method='sobol',minimizer_kwargs={'method':'SLSQP','jac':dMinMax,'args':(model,-1,0)})
    MinV = shgo(MinMax, bnds, args=(model,-1,1), n=100, iters=1, sampling_method='sobol',minimizer_kwargs={'method':'SLSQP','jac':dMinMax,'args':(model,-1,1)})
    GlobRange = np.array([[MinP.fun,MinV.fun],[-MaxP.fun,-MaxV.fun]])
    GlobExtrem = GlobRange*(OutputRange[1] - OutputRange[0]) + OutputRange[0]

    print("Power range: {:.2f} - {:.2f} W".format(*GlobExtrem[:,0]))
    print("Unifortmity score range: {:.2f} - {:.2f}\n".format(*GlobExtrem[:,1]))

    DesPower = 500
    alpha = 1

    print("A minimum of {:.2f} W of power is required".format(DesPower))
    print("Power weighted by {:.3f}, Variation weighted by {:.3f}".format(alpha,1-alpha))

    # Define constraints
    DesPower_norm = ((DesPower - OutputRange[0])/(OutputRange[1]-OutputRange[0]))[0]
    con1 = {'type': 'ineq', 'fun': constraint,'jac':dconstraint, 'args':(model, DesPower_norm)}
    cnstr = ([con1])

    print()
    NbInit = 5
    rnd = np.random.uniform(0,1,size=(NbInit,4))
    OptScores = []
    for i, X0 in enumerate(rnd):
        OptScore = minimize(func, X0, args=(model, alpha), method='SLSQP',jac=dfunc, bounds=bnds, constraints=cnstr, options={'maxiter':100})
        OptScores.append(OptScore)

    Score = []
    tol = 0.001
    for Opt in OptScores:
        if not Opt.success:
            continue
        # print(Opt.x)
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







    # def GradLagr(model, x, mu, DesPower_norm, alpha):
    #     Sgrad = Grad1(model, x)
    #     Lgrads = -(alpha*Sgrad[0,:]-(1-alpha)*Sgrad[1,:]) + mu*Sgrad[0,:]
    #     mugrad = Constraint(model, x, DesPower_norm)
    #     return np.append(Lgrads,mugrad.detach().numpy())

    # ds = 0.001
    # lr=0.0005
    # alpha=1
    # mu=0.5
    #
    # DesPower = 500
    # DesPower_norm = ((DesPower - OutputRange[0])/(OutputRange[1]-OutputRange[0]))[0]
    #
    # print("Requiring a minimum power of {} W".format(DesPower))
    # print(DesPower_norm)
    #
    # Mag = np.linalg.norm
    #
    # # OldCoord = torch.tensor(PMaxGridCd, dtype=torch.float32)
    # # Output = model.predict(OldCoord)
    # # print(Output)
    # # Score = alpha*Output[0,:] + (1-alpha)*(1-Output[1,:])
    # # print(Output, Score)
    #
    # StartCds = grid[NNGrid[:,0] > DesPower_norm,:]
    # Coords = torch.tensor(StartCds, dtype=torch.float32)

    # print('Maximise PowerUniformity score')
    # for i in range(1):
    #     Sgrad = torch.tensor(Grad1(model, Coords), dtype=torch.float32)
    #     Coords = Coords - lr*(-(alpha*Sgrad[:,:,0]-(1-alpha)*Sgrad[:,:,1]))
    #
    #     mask = Coords < 0
    #     if mask.byte().any():  Coords[mask] = 0
    #     mask = Coords > 1
    #     if mask.byte().any(): Coords[mask] = 1
    #
    #     if i % 50 == 0:
    #         pred = model.predict(Coords)
    #         Score = alpha*pred[:,0] + (1-alpha)*(1-pred[:,1])
    #         # print("{}: {:.5f}".format(i,Score))
    #         print(Score)

    # if Constraint(model,OldCoord,DesPower_norm)<0:
    #     print("Minimise mag. lagrange function to satisfy constraint")
    #     for i in range(50):
    #         Mag = Mag(GradLagr(model,OldCoord,mu,DesPower_norm,alpha),axis=1)
    #         # Constraint is not satisfied so looking to find min point
    #         # of mangitude of lagrange partials
    #         G_grads = []
    #         for ipar in np.arange(4):
    #             FDp = OldCoord.clone()
    #             FDm = OldCoord.clone()
    #             FDp[ipar]+=ds
    #             FDm[ipar]-=ds
    #
    #             G_grad = (Mag(GradLagr(model,FDp,mu,DesPower_norm,alpha)) - Mag(GradLagr(model,FDm,mu,DesPower_norm,alpha)))/(2*ds)
    #             G_grads.append(G_grad)
    #
    #         mp = (Mag(GradLagr(model,FDp,mu+ds,DesPower_norm,alpha)) - Mag(GradLagr(model,FDp,mu-ds,DesPower_norm,alpha)))/(2*ds)
    #
    #         OldCoord = OldCoord - lr*torch.tensor(G_grads, dtype=torch.float32)
    #         mu = mu - lr*mp
    #         NewMag = Mag(GradLagr(model,OldCoord,mu,DesPower_norm,alpha))
    #         if i % 10 == 0:
    #             print("{}: {:.5f}".format(i, Mag(GradLagr(model,OldCoord,mu,DesPower_norm,alpha))))
    #         if Constraint(model,OldCoord,DesPower_norm)>=0:
    #             break



    # Satisfied = False
    # NbUnsat, NbSat = 0,0
    # for i in range(20000):
    #     if not Satisfied and Constraint(model,OldCoord,DesPower_norm)>=0:
    #         Satisfied = True
    #
    #     if not Satisfied:
    #         if NbUnsat == 0:
    #             print("Minimise mag. lagrange function to satisfy constraint")
    #             NbUnsat=1
    #         # Constraint is not satisfied so looking to find min point
    #         # of mangitude of lagrange partials
    #         G_grads = []
    #         for ipar in np.arange(4):
    #             FDp = OldCoord.clone()
    #             FDm = OldCoord.clone()
    #             FDp[ipar]+=ds
    #             FDm[ipar]-=ds
    #
    #             G_grad = (Mag(GradLagr(model,FDp,mu,DesPower_norm,alpha)) - Mag(GradLagr(model,FDm,mu,DesPower_norm,alpha)))/(2*ds)
    #             G_grads.append(G_grad)
    #
    #         mp = (Mag(GradLagr(model,FDp,mu+ds,DesPower_norm,alpha)) - Mag(GradLagr(model,FDp,mu-ds,DesPower_norm,alpha)))/(2*ds)
    #         print(Constraint(model,OldCoord,DesPower_norm))
    #         OldCoord = OldCoord - lr*torch.tensor(G_grads, dtype=torch.float32)
    #         mu = mu - lr*mp
    #
    #         if i % 1 == 0 :
    #             print("{}: {:.5f}".format(i, Mag(GradLagr(model,OldCoord,mu,DesPower_norm,alpha))))
    #             print(mu, Constraint(model,OldCoord,DesPower_norm))
    #     else:
    #         if NbSat == 0:
    #             print('Maximise PowerUniformity score')
    #             NbSat=1
    #         Sgrad = torch.tensor(Grad1(model, OldCoord), dtype=torch.float32)
    #         OldCoord = OldCoord - lr*(-(alpha*Sgrad[0,:]-(1-alpha)*Sgrad[1,:]))
    #         if any(OldCoord < 0): OldCoord[OldCoord < 0] = 0
    #         if any(OldCoord > 1): OldCoord[OldCoord > 1] = 1
    #
    #         if i % 50 == 0 and i!=0:
    #             pred = model.predict(OldCoord)
    #             Score = alpha*pred[0] + (1-alpha)*(1-pred[1])
    #             print(i, Score)

        # if i % 100 == 0 and i!=0:
        #     pred = model.predict(OldCoord)
        #     print(i, alpha*pred[0] + (1-alpha)*(1-pred[1]), OldCoord)

    # optimum = (model.predict(OldCoord)*(OutputRange[1] - OutputRange[0]) + OutputRange[0]).detach().numpy()
    # print(optimum)
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

    # def grads(model, x, mu, DesPower_norm):
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
    #     L_grads.append(model.predict(x)[0]-DesPower_norm)
    #
    #     return L_grads
    #
    # print(np.linalg.norm(grads(model,OldCoord,mu,DesPower_norm)))
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
