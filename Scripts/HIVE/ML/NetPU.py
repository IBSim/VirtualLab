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

    def predict(self,xn,device,GPU=False):
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

# unstructured data to structured
def to_structured(rxyz, data):
    # find grid locations
    rxyz_np = rxyz.detach().numpy()
    grid_loc = []
    grid_loc_dict = []
    for ipar in np.arange(4):
        grid_loc.append(np.unique(rxyz_np[:, ipar]))
        grid_loc_dict.append({})
        for loc, val in enumerate(grid_loc[-1]):
            grid_loc_dict[-1][val] = loc

    # fill grid values
    grid_data = np.zeros((len(grid_loc[0]), len(grid_loc[1]),
                          len(grid_loc[2]), len(grid_loc[3]),
                          data.shape[-1]), dtype=np.float32)
    for i, irxyz in enumerate(rxyz_np):
        loc = [0, 0, 0, 0]
        for ipar in np.arange(4):
            loc[ipar] = grid_loc_dict[ipar][irxyz[ipar]]
        grid_data[loc[0], loc[1], loc[2], loc[3], :] = data[i, :].detach().numpy()
    return grid_loc, grid_data

# compute objective and sort
def compute_sort(model, rxyz, range, rxyz_grid=None, GPU=False):
    # constrain range
    if rxyz_grid is not None:
        rxyz_use = rxyz.detach().numpy().copy()
        for ipar in np.arange(4):
            loc = np.where((rxyz_use[:, ipar] >= rxyz_grid[ipar][0]) *
                           (rxyz_use[:, ipar] <= rxyz_grid[ipar][-1]))[0]
            rxyz_use = rxyz_use[loc, :]
        rxyz_use = torch.from_numpy(rxyz_use)
    else:
        rxyz_use = rxyz

    # sort objective
    obj = compute_objective(model, rxyz_use, range, GPU)
    arg = torch.argsort(obj, descending=True)
    return obj[arg], rxyz_use[arg, :]

# parameter update based on gradient descent
def gradient_update(model, rxyz, range, fintie_steps=[.001, 0.000005, 0.000005, 0.0000005],
                    lr=.0000001, GPU=False):
    rxyz_updated = rxyz.clone()
    for ipar in np.arange(4):
        # gradient by finite difference
        rxyz_dpar_p = rxyz.clone()
        rxyz_dpar_m = rxyz.clone()
        rxyz_dpar_p[:, ipar] += fintie_steps[ipar]
        rxyz_dpar_m[:, ipar] -= fintie_steps[ipar]
        dobj_dpar = (compute_objective(model, rxyz_dpar_p, range, GPU) -
                     compute_objective(model, rxyz_dpar_m, range, GPU)) / (fintie_steps[ipar] * 2)
        # update by learning rate
        rxyz_updated[:, ipar] += dobj_dpar * lr
    return rxyz_updated


# plot structured data
def plot_structured(rxyz_grid, data_grid, data_index, plot_norm=1, title=''):
    fig, ax = plt.subplots(nrows=len(rxyz_grid[0]), ncols=len(rxyz_grid[3]), dpi=200, figsize=(4,6))
    for iR, R in enumerate(rxyz_grid[0]):
        for iZ, Z in enumerate(rxyz_grid[3]):
            ax[iR, iZ].imshow(np.abs(data_grid[iR, :, :, iZ, data_index]),
                              vmin=0, vmax=plot_norm, origin='lower')
            ax[iR, iZ].set_xticks([])
            ax[iR, iZ].set_yticks([])
    for iR, R in enumerate(rxyz_grid[0]):
        ax[iR, 0].set_ylabel('%.1f' % R, fontsize=6)
    for iZ, Z in enumerate(rxyz_grid[3]):
        ax[-1, iZ].set_xlabel('%.4f' % Z, fontsize=6)
    fig.suptitle(title, fontsize=10, y=.08)
    plt.show()

# define the objective function
def objective(mu_sigma):
    return mu_sigma[:, 0] - mu_sigma[:, 1]

# compute objective by model
def compute_objective(model, rxyz, range, GPU=False):
    mu_sigma = model.predict_denorm(rxyz, *range)
    return objective(mu_sigma)

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

    NNInput, NNOutput = 4,2

    # input: (r, x, y, z)
    train_rxyz = torch.from_numpy(TrainData[:,:4])
    test_rxyz  = torch.from_numpy(TestData[:,:4])

    # output: (mu, sigma)
    train_mu_sigma = torch.from_numpy(TrainData[:,4:])
    test_mu_sigma  = torch.from_numpy(TestData[:,4:])

    # normalized data for training
    min_rxyz = train_rxyz.min(axis=0).values
    max_rxyz = train_rxyz.max(axis=0).values

    train_rxyz_norm = ((train_rxyz[None, :] - min_rxyz) / (max_rxyz - min_rxyz))[0]
    test_rxyz_norm  = ((test_rxyz[None, :] - min_rxyz) / (max_rxyz - min_rxyz))[0]

    min_mu_sigma = train_mu_sigma.min(axis=0).values
    max_mu_sigma = train_mu_sigma.max(axis=0).values
    train_mu_sigma_norm = ((train_mu_sigma[None, :] - min_mu_sigma) / (max_mu_sigma - min_mu_sigma))[0]
    test_mu_sigma_norm  = ((test_mu_sigma[None, :] - min_mu_sigma) / (max_mu_sigma - min_mu_sigma))[0]

    ModelFile = '{}/model_{}.h5'.format(VL.ML_DIR, MLdict.Name)

    range = [min_rxyz, max_rxyz, min_mu_sigma, max_mu_sigma]

    if MLdict.NewModel:
        torch.manual_seed(123)
        # create model instance
        model = NetPU()

        # train the model
        model.train()

        history, epoch_stop = train(model, train_rxyz_norm, train_mu_sigma_norm,
        		            test_rxyz_norm, test_mu_sigma_norm, device,
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

    # prediction
    pred_train_mu_sigma = model.predict_denorm(train_rxyz, min_rxyz, max_rxyz, min_mu_sigma, max_mu_sigma)
    pred_test_mu_sigma  = model.predict_denorm(test_rxyz, min_rxyz, max_rxyz, min_mu_sigma, max_mu_sigma)

    # error
    train_error = (pred_train_mu_sigma - train_mu_sigma) / train_mu_sigma
    test_error  = (pred_test_mu_sigma - test_mu_sigma) / test_mu_sigma

    # plot error histograms
    fig,ax = plt.subplots(2, 2)
    ax[0,0].hist(train_error[:, 0].detach().numpy(),bins=20)
    ax[0,0].set_title('Power_train')
    ax[0,0].set_xlabel('Error')
    ax[0,0].set_ylabel('Count')
    Xlim = max(np.absolute(ax[0,0].get_xlim()))
    ax[0,1].hist(test_error[:, 0].detach().numpy(),bins=20)
    ax[0,1].set_title('Power_test')
    ax[0,1].set_xlabel('Error')
    ax[0,1].set_ylabel('Count')
    Xlim = max(max(np.absolute(ax[0,1].get_xlim())),Xlim)
    ax[1,0].hist(train_error[:, 1].detach().numpy(),bins=20)
    ax[1,0].set_title('Uniform_train')
    ax[1,0].set_xlabel('Error')
    ax[1,0].set_ylabel('Count')
    Xlim = max(max(np.absolute(ax[1,0].get_xlim())),Xlim)
    ax[1,1].hist(test_error[:, 1].detach().numpy(),bins=20)
    ax[1,1].set_title('Uniform_test')
    ax[1,1].set_xlabel('Error')
    ax[1,1].set_ylabel('Count')
    Xlim = max(max(np.absolute(ax[1,1].get_xlim())),Xlim)

    plt.setp(ax, xlim=(-Xlim,Xlim))
    # plt.show()

    with torch.no_grad():
        NNtrain_mu_sigma_norm = model(train_rxyz_norm)
        NNtest_mu_sigma_norm = model(test_rxyz_norm)

    trainSqE = (NNtrain_mu_sigma_norm - train_mu_sigma_norm).detach().numpy()**2
    trainMSE = np.sum(trainSqE,axis=0)/trainSqE.shape[0]
    print("Train errors\nPower:{}\nUniformity:{}".format(trainMSE[0],trainMSE[1]))

    testSqE = (NNtest_mu_sigma_norm - test_mu_sigma_norm).detach().numpy()**2
    testMSE = np.sum(testSqE,axis=0)/testSqE.shape[0]
    print("Test errors\nPower:{}\nUniformity:{}".format(testMSE[0],testMSE[1]))
    # import time
    # st = time.time()

    disc = np.linspace(0, 1, 10)
    grid = np.meshgrid(disc, disc, disc, disc,indexing='ij')
    grid = np.moveaxis(np.array(grid),0,-1) #grid point is now the last axis
    ndim = grid.ndim - 1

    grid_1 = grid.reshape([disc.shape[0]**ndim,ndim])
    grid_1 = torch.tensor(grid_1, dtype=torch.float32)

    with torch.no_grad():
        vals = model.predict(grid_1,device,GPU=False)

    npvals = vals.detach().numpy()
    npvals = npvals.reshape([disc.shape[0]]*ndim+[npvals.shape[1]])

    # Power
    print_count = 5
    print('')
    sort1 = np.argsort(-npvals[:,:,:,:,0],axis=None)
    sort1 = np.unravel_index(sort1,npvals.shape[:-1])
    Powsrt, loc1 = npvals[sort1][:,0], grid[sort1]
    for pow,coord in zip(Powsrt[:print_count],loc1):
        print(coord, '==>', pow)

    print('')
    sort2 = np.argsort(npvals[:,:,:,:,1],axis=None)
    sort2 = np.unravel_index(sort2,npvals.shape[:-1])
    Unisrt, loc2 = npvals[sort2][:,1], grid[sort2]
    for pow,coord in zip(Unisrt[:print_count],loc2):
        print(coord, '==>', pow)


    #
    #
    #
    # rxyz = np.meshgrid(disc, disc, disc, disc)
    # rxyz = [rxyz[0].reshape(-1), rxyz[1].reshape(-1), rxyz[2].reshape(-1), rxyz[3].reshape(-1)]
    # rxyz = torch.tensor(np.array(rxyz).T, dtype=torch.float32)
    #
    # with torch.no_grad():
    #     vals = model.predict(rxyz,device,GPU=False)
    #
    # # Power
    # print('')
    # sort1 = torch.argsort(vals[:,0], descending=True)
    # Powsrt, loc1 = vals[sort1,0],rxyz[sort1,:]
    # for i in np.arange(print_count):
    #     print(loc1[i].detach().numpy(), '==>', Powsrt[i].detach().numpy())
    #
    # print('')
    # sort2 = torch.argsort(vals[:,1], descending=False)
    # Unisrt, loc2 = vals[sort2,1],rxyz[sort2,:]
    # for i in np.arange(print_count):
    #     print(loc2[i].detach().numpy(), '==>', Unisrt[i].detach().numpy())


    # val,loc = obj[arg],rxyz[arg,:]
    # print_count = 10
    # for i in np.arange(print_count):
    #     print(loc[i].detach().numpy(), '==>', val[i].detach().numpy())

    #
    # trainSqE = (pred_train_mu_sigma - train_mu_sigma).detach().numpy()**2
    # trainMSE = np.sum(trainSqE,axis=0)/trainSqE.shape[0]
    # print("Train errors\nPower:{}\nUniformity:{}".format(trainMSE[0],trainMSE[1]))
    #
    # testSqE = (pred_test_mu_sigma - test_mu_sigma).detach().numpy()**2
    # testMSE = np.sum(testSqE,axis=0)/testSqE.shape[0]
    # print("Test errors\nPower:{}\nUniformity:{}".format(testMSE[0],testMSE[1]))
