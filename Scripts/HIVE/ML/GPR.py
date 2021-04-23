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
import torch.utils.data as Data
import gpytorch

from Scripts.Common.VLFunctions import MeshInfo
from Functions import Uniformity2 as UniformityScore

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=4))
#        self.covar_module = gpytorch.kernels.RBFKernel(lengthscale_prior=None)
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def Gradient(self, x):
        x.requires_grad=True
        with gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self(x))
            grads = torch.autograd.grad(pred.mean.sum(), x)[0]
            return grads

def Plot4D(Slice, MajorN, MinorN):
    InTag = ['x','y','z','r']

    AxMin, AxMaj = [], []
    for i, cmp in enumerate(InTag):
        if Slice.find(cmp) == -1: AxMaj.append(i)
        else : AxMin.append(i)

    # Discretisation
    DiscMin = np.linspace(0+0.5*1/MinorN,1-0.5*1/MinorN,MinorN)
    DiscMaj = np.linspace(0,1,MajorN)
    # DiscMaj = np.linspace(0+0.5*1/MajorN,1-0.5*1/MajorN,MajorN)
    disc = [DiscMaj]*4
    disc[AxMin[0]] = disc[AxMin[1]] = DiscMin
    grid = np.meshgrid(*disc, indexing='ij')
    grid = np.moveaxis(np.array(grid),0,-1) #grid point is now the last axis
    ndim = grid.ndim - 1
    # unroll grid
    _disc = [dsc.shape[0] for dsc in disc]
    grid = grid.reshape([np.prod(_disc),ndim])
    return grid

def DataScale(data,const,scale):
    '''
    This function scales n-dim data to a specific range.
    data: N-darray or scalar
    const: N-darray or scalar
    scale: N-darray or scalar
    Examples:
     - Normalising data:
        const=mean, scale=stddev
     - [0,1] range:
        const=min, scale=max-min
    '''
    return (data - const)/scale

def DataRescale(data,const,scale):
    '''
    This function scales data back to original range.
    data: N-darray or scalar
    const: N-darray or scalar
    scale: N-darray or scalar
    '''
    return data*scale + const

def MinMax(X, model, sign, Ix):
    '''
    Function which can be called to find min/max of power/uniform
    Sign = 1 => Max Value, Sign = -1 => Min Value
    Ix = 0 corresponds to power, Ix = 1 correspons to uniformity
    '''
    X = torch.tensor(np.atleast_2d(X),dtype=torch.float32)

    Pred = model(X).mean.detach().numpy()

    return -sign*Pred

def dMinMax(X, model, sign, Ix):
    '''
    Derivative of function MinMax
    '''
    # Grad = Gradold(model, X)[Ix,:]
    X = torch.tensor(np.atleast_2d(X),dtype=torch.float32)
    Grad = model.Gradient(X)[0]
    dMM = Grad

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

def Single(VL, MLdict):
    ML = MLdict["Parameters"]

    # File where all data is stored
    DataFile = "{}/Data.hdf5".format(VL.ML_DIR)
    NbTorchThread = getattr(ML,'NbTorchThread',1)
    torch.set_num_threads(NbTorchThread)
    torch.manual_seed(getattr(ML,'Seed',100))

    # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')

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
    Train_x, Train_y = TrainData[:,:4], (TrainData[:,4:])
    Test_x, Test_y = TestData[:,:4], (TestData[:,4:])

    # Scale input data to [0,1] (based on training data)
    InputRange = np.array([Train_x.min(axis=0),Train_x.max(axis=0) - Train_x.min(axis=0)])
    Train_x_scale = DataScale(Train_x,*InputRange)
    Test_x_scale = DataScale(Test_x,*InputRange)

    # Scale output to mean 0 and stddev 1 (based on training data)

    OutputRange = np.array([np.mean(Train_y,axis=0),np.std(Train_y,axis=0)])
    Train_y_scale = DataScale(Train_y,*OutputRange)
    Test_y_scale = DataScale(Test_y,*OutputRange)

    # bl = (Train_x_scale[:,1]==0.5)*(Train_x_scale[:,2]==0)*(Train_x_scale[:,3]==0.5)
    # Train_x_scale = Train_x_scale[bl,0]
    # Train_y_norm = Train_y_norm[bl]

    Train_x_tf = torch.from_numpy(Train_x_scale)
    Train_y_tf = torch.from_numpy(Train_y_scale)
    Test_x_tf = torch.from_numpy(Test_x_scale)
    Test_y_tf = torch.from_numpy(Test_y_scale)

    #==================================================================

    GPU = getattr(ML,'GPU', False)
    lr = getattr(ML,'lr', 0.0001)
    PrintEpoch = getattr(ML,'PrintEpoch',100)
    ConvCheck = getattr(ML,'ConvCheck',100)
    training_iter=300

    # Training model for power
    PowerTrain = Train_y_tf[:,0]
    PowerTest = Test_y_tf[:,0]

    ModelFile = '{}/Power.pth'.format(MLdict["CALC_DIR"]) # File model will be saved to/loaded from
    # initialize likelihood and model
    PowerLH = gpytorch.likelihoods.GaussianLikelihood()
    # likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Interval(0,1e-2))
    Power = ExactGPModel(Train_x_tf, PowerTrain, PowerLH)

    if ML.Train:
        # Find optimal model hyperparameters
        Power.train()
        PowerLH.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(Power.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(PowerLH,Power)

        TrainLoss = []
        for i in range(training_iter):
            optimizer.zero_grad() # Zero gradients from previous iteration
            # Output from model
            output = Power(Train_x_tf)
            # Calc loss and backprop gradients
            loss = -mll(output, PowerTrain)
            loss.backward()

            optimizer.step()

            if i % 10 == 0:
                print(i+1,loss.item(),Power.covar_module.base_kernel.lengthscale,Power.likelihood.noise.item(),Power.covar_module.outputscale)
            # if i % 10 == 0:
            #     print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.5f outputscale:%.3f' % (
            #     i + 1, training_iter, loss.item(),
            #     model.covar_module.base_kernel.lengthscale.item(),
            #     model.likelihood.noise.item(),
            #     model.covar_module.outputscale
            #     ))

            TrainLoss.append(loss.item())

            # if i>1:
            #     if abs(TrainLoss[-1] - TrainLoss[-2]) < 1e-6: break

        torch.save(Power.state_dict(), ModelFile)

        plt.figure()
        plt.plot(list(range(1,len(TrainLoss)+1)),TrainLoss, label='Training')
        plt.legend()
        plt.savefig("{}/Convergence.png".format(MLdict["CALC_DIR"]))
        plt.close()

        Power.eval()
        PowerLH.eval()

        with torch.no_grad():
            TrainPred = PowerLH(Power(Train_x_tf))
            TrainMSE = np.mean((OutputRange[1,0]*(TrainPred.mean-PowerTrain).numpy())**2)

            TestPred = PowerLH(Power(Test_x_tf))
            TestMSE = np.mean((OutputRange[1,0]*(TestPred.mean-PowerTest).numpy())**2)
            print(TrainMSE,TestMSE)

    else:
        state_dict = torch.load(ModelFile)
        Power.load_state_dict(state_dict)

    Power.eval()
    PowerLH.eval()

    model = Power

    if 0:
        Slice = 'xy'
        Res = 'Power' # 'Uniformity'
        Component = 'gradmag'
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


        grid_tf = torch.tensor(grid, dtype=torch.float32)
        # decide what component of Res to print - gradient (magnitude or individual) or the value
        if Component.lower().startswith('grad'):
            imres = model.Gradient(grid_tf).detach().numpy()
            if Component.lower() == 'gradmag':
                imres = np.linalg.norm(imres, axis=1)
            else:
                # letter after grad must be x,y,z or r
                ax = InTag.index(Component[-1].lower())
                imres = imres[:,ax]
        else :
            with torch.no_grad():
                imres = model(grid_tf).mean.detach().numpy()
        imres = DataRescale(imres,*OutputRange[:,0])

        # min and max value for global colour bar
        IMMin, IMMax = imres.min(axis=0), imres.max(axis=0)

        imres = imres.reshape(_disc+[*imres.shape[1:]])
        # scaled vales for major axis
        arr1 = DataRescale(DiscMaj,*InputRange[:,AxMaj[0]])
        arr2 = DataRescale(DiscMaj,*InputRange[:,AxMaj[1]])

        PlotTrain=False
        df = DiscMaj[1] - DiscMaj[0]
        # errmin, errmax = train_error.min(axis=0)[ResAx], train_error.max(axis=0)[ResAx]
        # errbnd = max(abs(errmin),abs(errmax))

        fig, ax = plt.subplots(nrows=arr1.size, ncols=arr2.size, sharex=True, sharey=True, dpi=200, figsize=(12,9))
        ax = np.atleast_2d(ax)

        fig.subplots_adjust(right=0.8)
        for it1,(dsc1,nb1) in enumerate(zip(DiscMaj,arr1)):
            _it1 = -(it1+1)
            ax[_it1, 0].set_ylabel('{:.4f}'.format(nb1), fontsize=12)
            for it2,(dsc2,nb2) in enumerate(zip(DiscMaj,arr2)):
                sl = [slice(None)]*len(InTag)
                sl[AxMaj[0]],sl[AxMaj[1]]  = it1, it2

                Im = ax[_it1,it2].imshow(imres[tuple(sl)].T, cmap = 'coolwarm', vmin=IMMin, vmax=IMMax, origin='lower',extent=(0,1,0,1))
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
        plt.show()
        # plt.savefig("{}/NNPlot.png".format(MLdict["CALC_DIR"]))
        plt.close()


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
            MaxPower_vals = model(torch.tensor(MaxPower_cd, dtype=torch.float32))
        MaxPower_val = DataRescale(MaxPower_vals.mean,*OutputRange[:,0])
        print(MaxPower_vals.stddev)
        MaxPower_cd = DataRescale(MaxPower_cd, *InputRange)
        print("{:8}{:12}{:12}".format('Epoch','Train_loss','Val_loss'))
        print("    {:7}{:7}{:7}{:11}{:9}{:8}".format('x','y','z','r','Power','Variation'))
        for coord, val in zip(MaxPower_cd,MaxPower_val):
            print("({:.4f},{:.4f},{:.4f},{:.4f}) ---> {:.2f} W".format(*coord, val))
        print()

    # _tst = torch.tensor(np.atleast_2d([[0.5,0.5,0.5,0.5],[0.5,0.6,0.5,0.5]]), dtype=torch.float32)
    # grd = model.GPRGrad(_tst)
    # print(grd)


    # test_pred = model(Test_x_tf)
    # Test_diff = Train_y_stdev*(test_pred.mean - Test_y_tf).detach().numpy().T
    #
    # test_stddev = Train_y_stdev*test_pred.stddev.detach().numpy()
    # test_stddev_max = test_stddev.max()
    #
    # plt.figure()
    # plt.scatter(test_stddev,np.abs(Test_diff))
    # for i in range(1,4):
    #     plt.plot([0,test_stddev_max],[0,i*test_stddev_max],label='{} stdev from mean'.format(i))
    # plt.xlabel('Uncertinity')
    # plt.ylabel('Absolute error')
    # plt.legend()
    # plt.show()
    # Test_loss = np.mean((Test_diff)**2)
    # print(Test_loss)
