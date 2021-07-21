import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import minimize, differential_evolution, shgo, basinhopping
from types import SimpleNamespace as Namespace
import torch
import gpytorch

from Scripts.Common.VLFunctions import MeshInfo
from Functions import Uniformity2 as UniformityScore, DataScale, DataRescale, FuncOpt
from PreAster.PreHIVE import ERMES

def InputParameters():
    pass

def Single(VL, MLdict):
    ML = MLdict["Parameters"]

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

        np.save("{}/TrainData".format(MLdict["CALC_DIR"]),TrainData)

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

        np.save("{}/TestData".format(MLdict["CALC_DIR"]),TestData)

        MLData.close()

    else:
        TrainData = np.load("{}/TrainData.npy".format(MLdict["CALC_DIR"]))
        TestData = np.load("{}/TestData.npy".format(MLdict["CALC_DIR"]))

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

    #======================================================================

    Train_P_tf,Train_V_tf = Train_y_tf[:,0],Train_y_tf[:,1]
    Test_P_tf,Test_V_tf = Test_y_tf[:,0],Test_y_tf[:,1]

    if getattr(ML,'Noise',True):
        PowerLH = gpytorch.likelihoods.GaussianLikelihood()
        VarLH = gpytorch.likelihoods.GaussianLikelihood()
    else:
        # Noise can't be zero in gpytorch so set it to very small value
        sig = 0.00001*torch.ones(Train_x_tf.shape[0])
        PowerLH = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(sig)
        VarLH = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(sig)

    # If kernel is 'Matern' then nu must be specified in Parameters
    options = {}
    if hasattr(ML,'Nu'): options['nu']=ML.Nu

    Power = ExactGPmodel(Train_x_tf, Train_P_tf, PowerLH,
                    ML.Kernel,options)
    Variation = ExactGPmodel(Train_x_tf, Train_V_tf, VarLH,
                    ML.Kernel,options)

    if ML.Train:
        # Power
        lr = getattr(ML,'lr', 0.01)
        MLdict['Data']['MSE'] = MSEvals = {}
        print('Training Power')
        Conv_P,MSE_P = Power.Training(PowerLH,ML.Iterations,lr=lr,)
        print()
        ModelFile = '{}/Power.pth'.format(MLdict["CALC_DIR"]) # File model will be saved to/loaded from
        torch.save(Power.state_dict(), ModelFile)

        if MSE_P:
            plt.figure()
            plt.plot(np.array(MSE_P)*OutputScaler[1,0]**2)
            plt.savefig("{}/MSE_Power.png".format(MLdict["CALC_DIR"]))
            plt.close()

        with torch.no_grad(), gpytorch.settings.max_cholesky_size(1500), gpytorch.settings.debug(False):
            Power_test = MSE(Power(Test_x_tf).mean.numpy(), Test_y_scale[:,0])
            Power_train = MSE(Power(Train_x_tf).mean.numpy(), Train_y_scale[:,0])
        Power_test *= OutputScaler[1,0]**2
        Power_train *= OutputScaler[1,0]**2
        print("Train MSE: {}\nTest MSE: {}\n".format(Power_train,Power_test))

        print(Power_test,Power_train)
        MSEvals["Power_Test"] = Power_test
        MSEvals["Power_Train"] = Power_train

        # Variation
        print('Training Variation')
        Conv_V,MSE_V = Variation.Training(VarLH,ML.Iterations,lr=lr,)
        print()
        ModelFile = '{}/Variation.pth'.format(MLdict["CALC_DIR"]) # File model will be saved to/loaded from
        torch.save(Variation.state_dict(), ModelFile)

        if MSE_V:
            plt.figure()
            plt.plot(MSE_V)
            plt.savefig("{}/MSE_Variation.png".format(MLdict["CALC_DIR"]))
            plt.close()

        with torch.no_grad(), gpytorch.settings.max_cholesky_size(1500), gpytorch.settings.debug(False):
            Vari_test = MSE(Variation(Test_x_tf).mean.numpy(), Test_y_scale[:,1])
            Vari_train = MSE(Variation(Train_x_tf).mean.numpy(), Train_y_scale[:,1])
        Vari_test  *= OutputScaler[1,1]**2
        Vari_train *= OutputScaler[1,1]**2
        print("Train MSE: {}\nTest MSE: {}\n".format(Vari_train,Vari_test))
        MSEvals["Variation_Test"] = Vari_test
        MSEvals["Variation_Train"] = Vari_train

        with torch.no_grad(), gpytorch.settings.max_cholesky_size(1500), gpytorch.settings.debug(False):
            Test_MSE_P = MSE(Power(Test_x_tf).mean.numpy(), Test_y_scale[:,0])
            Test_MSE_P *= OutputScaler[1,0]**2 #scale MSE to correct range

            Train_MSE_P = MSE(Power(Train_x_tf).mean.numpy(), Train_y_scale[:,0])
            Train_MSE_P *= OutputScaler[1,0]**2 #scale MSE to correct range

        plt.figure()
        plt.plot(Conv_V, label='Variation')
        plt.plot(Conv_P,label='Power')
        plt.legend()
        plt.savefig("{}/Convergence.png".format(MLdict["CALC_DIR"]))
        plt.close()

    else:
        # Power
        state_dict_P = torch.load('{}/Power.pth'.format(MLdict["CALC_DIR"]))
        Power.load_state_dict(state_dict_P)
        PowerLH.eval(); Power.eval()

        Variation
        state_dict_V = torch.load('{}/Variation.pth'.format(MLdict["CALC_DIR"]))
        Variation.load_state_dict(state_dict_V)
        Variation.eval(); VarLH.eval()

    if getattr(ML,'Input',None) != None:
        with torch.no_grad():
            x_scale = DataScale(np.atleast_2d(ML.Input),*InputScaler)
            x_scale = torch.tensor(x_scale, dtype=torch.float32)
            y = Power(x_scale)
            y_mean = DataRescale(y.mean.numpy(),*OutputScaler[:,0])
            y_stddev = DataRescale(y.stddev,0,OutputScaler[1,0])
            print(y_mean)
            print(y_stddev)
            # print(y_mean)
            MLdict['Output'] = y_mean.tolist()
            MLdict['Output_Var'] = y_stddev.tolist()

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

    Optima = FuncOpt(GPR_Opt, NbInit, bnds, args=[Power],
                    find='max', tol=0.01, order='decreasing')
    MaxPower_cd,MaxPower_val,MaxPower_grad = Optima
    # for _MaxPower_cd in MaxPower_cd:
    #     with torch.no_grad():
    #         MaxPower_cd_tf = torch.tensor(np.atleast_2d(_MaxPower_cd), dtype=torch.float32)
    #         GPR_Out = Power(MaxPower_cd_tf)
    #     _MaxPower_cd = DataRescale(_MaxPower_cd, *InputScaler)
    #     _MaxPower_val = DataRescale(GPR_Out.mean.numpy(),*OutputScaler[:,0])
    #     print(_MaxPower_cd, _MaxPower_val)

    with torch.no_grad():
        MaxPower_cd_tf = torch.tensor(MaxPower_cd, dtype=torch.float32)
        MPPower = Power(MaxPower_cd_tf)
        MPVar = Variation(MaxPower_cd_tf)
    MaxPower_cd = DataRescale(MaxPower_cd, *InputScaler)
    MPMean = np.vstack((MPPower.mean.numpy(),MPVar.mean.numpy())).T
    MPStd = np.vstack((MPPower.stddev.detach(),MPVar.stddev.detach())).T

    MaxPower_val = DataRescale(MPMean,*OutputScaler)

    MaxPower_std = DataRescale(MPStd,0,OutputScaler[1,:])
    print("    {:7}{:7}{:7}{:11}{:9}".format('x','y','z','r','Power'))
    for coord, val, std in zip(MaxPower_cd,MaxPower_val,MaxPower_std):
        print("({0:.4f},{1:.4f},{2:.4f},{3:.4f}) ---> {4:.2f} W ({6}) {5:.2f} ({7})".format(*coord,*val,*std))
    print()

    MLdict["Data"]['MaxPower'] = MaxPower = {'x':MaxPower_cd[0],'y':MaxPower_val[0,0]}

    if ML.MaxPowerOpt.get('Verify',True):
        print("Checking results at optima\n")
        ERMESMaxPower = '{}/MaxPower.rmed'.format(MLdict["CALC_DIR"])
        EMParameters = Param_ERMES(*MaxPower_cd[0],ERMES_Parameters)
        RunERMES = ML.MaxPowerOpt.get('NewSim',True)

        JH_Vol, Volumes, Elements, JH_Node = ERMES(VL,MeshFile,ERMESMaxPower,
                                                EMParameters, MLdict["TMP_CALC_DIR"],
                                                RunERMES, GUI=0)
        Watts = JH_Vol*Volumes
        # Power & Uniformity
        Power = np.sum(Watts)
        JH_Node /= 1000**2
        Uniformity = UniformityScore(JH_Node,ERMESMaxPower)
        print(MaxPower_val)
        print("Anticipated power at optimum configuration: {0:.2f} W".format(MaxPower_val[0,0]))
        print("Actual power at optimum configuration: {:.2f} W\n".format(Power))

        MaxPower["target"] = Power


    if hasattr(ML,'DesPowerOpt'):
        if ML.DesPowerOpt['Power'] >= MaxPower_val[0]:
            print('DesiredPower greater than power available.\n')
        else:
            print("Locating optimum configuration(s) for maximum uniformity, ensuring power >= {} W".format(ML.DesPowerOpt['Power']))
            NbInit = ML.DesPowerOpt.get('NbInit',20)

            # constraint to ensure power requirement is met
            DesPower_norm = DataScale(ML.DesPowerOpt['Power'], *OutputScaler[:,0]) #scale power
            con1 = {'type': 'ineq', 'fun': constraint, 'jac':dconstraint, 'args':(Power, DesPower_norm)}

            Optima = FuncOpt(GPR_Opt, NbInit, bnds, args=[Variation],
                            find='min', tol=0.01, order='increasing',
                            constraints=con1, options={'maxiter':100})

            OptVar_cd,OptVar_val,OptVar_grad = Optima
            with torch.no_grad():
                OptVar_tf = torch.tensor(OptVar_cd, dtype=torch.float32)
                _P,_V = Power(OptVar_tf), Variation(OptVar_tf)
                PV = _P.mean.numpy(),_V.mean.numpy()

            OptVar_val = DataRescale(np.array(PV).T,*OutputScaler)
            OptVar_cd = DataRescale(OptVar_cd, *InputScaler)

            print("    {:7}{:7}{:7}{:11}{:9}{:8}".format('x','y','z','r','Power','Variation'))
            for coord, val in zip(OptVar_cd, OptVar_val):
                print("({:.4f},{:.4f},{:.4f},{:.4f}) ---> {:.2f} W, {:.3f}".format(*coord, *val))
            print()

            if ML.DesPowerOpt.get('Verify',True):
                print("Checking results at optima\n")
                ERMESRes = '{}/MinVar_{}.rmed'.format(MLdict["CALC_DIR"],ML.DesPowerOpt['Power'])
                RunERMES = ML.DesPowerOpt.get('NewSim',True)
                EMParameters = Param_ERMES(*OptVar_cd[0],ERMES_Parameters)

                JH_Vol, Volumes, Elements, JH_Node = ERMES(VL,MeshFile,ERMESRes,
                                                    EMParameters, MLdict["TMP_CALC_DIR"],
                                                    RunERMES, GUI=0)

                Watts = JH_Vol*Volumes
                # Power & Uniformity
                Power = np.sum(Watts)
                JH_Node /= 1000**2
                Uniformity = UniformityScore(JH_Node,ERMESRes)

                print("Anticipated at optimum configuration:\nPower: {:.2f} W\n"\
                      "Uniformity: {}".format(*OptVar_val[0]))
                print("Actual values at optimum configuration:\nPower: {:.2f} W\n"\
                      "Uniformity: {}".format(Power,Uniformity))


def MSE(Predicted,Target):
    sqdiff = (Predicted - Target)**2
    return np.mean(sqdiff)


class ExactGPmodel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel,options={}):
        super(ExactGPmodel, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()

        if kernel.lower() in ('rbf'):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=4))
        if kernel.lower() in ('matern'):
            nu = options.get('nu',2.5)
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=nu,ard_num_dims=4))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def Training(self,LH,Iterations=1000, lr=0.01,test=None, Print=50, ConvCheck=50,**kwargs):

        self.train()
        LH.train()
        # Use the adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)  # Includes GaussianLikelihood parameters
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(LH,self)

        ConvAvg = float('inf')
        Convergence = []
        TrainMSE, TestMSE = [],[]
        print("Iteration      Loss    Lengthscale    Noise    OutputScale")
        for i in range(Iterations):
            optimizer.zero_grad() # Zero gradients from previous iteration
            # Output from model
            with gpytorch.settings.max_cholesky_size(1500):
                output = self(self.train_inputs[0])
                # Calc loss and backprop gradients
                loss = -mll(output, self.train_targets)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                if i==0 or (i+1) % Print == 0:
                    print("{}    {}    {}    {}    {}".format(i+1, loss.numpy(),
                                                self.covar_module.base_kernel.lengthscale.numpy()[0],
                                                self.likelihood.noise.numpy()[0],
                                                self.covar_module.outputscale.numpy()))
                    # print(i+1, loss.item(),self.covar_module.base_kernel.lengthscale,
                    #         self.covar_module.outputscale)
                Convergence.append(loss.item())

            if test and i%50==0:
                with torch.no_grad(), gpytorch.settings.max_cholesky_size(1500),gpytorch.settings.fast_pred_var():
                    self.eval(); LH.eval()
                    x,y = test
                    pred = LH(self(x))
                    MSE = np.mean(((pred.mean-y).numpy())**2)
                    print("MSE",MSE)
                    TestMSE.append(MSE)
                    self.train();LH.train()
                    if (i+1) % ConvCheck == 0:
                        print(MSE)

            if (i+1) % ConvCheck == 0:
                Avg = np.mean(Convergence[-ConvCheck:])
                if Avg > ConvAvg:
                    print("Training terminated due to convergence")
                    break
                ConvAvg = Avg


        self.eval()
        LH.eval()
        return Convergence,TestMSE

    def Gradient(self, x):
        x.requires_grad=True
        with gpytorch.settings.fast_pred_var():
            # pred = self.likelihood(self(x))
            pred = self(x)
            grads = torch.autograd.grad(pred.mean.sum(), x)[0]
            return grads

def Param_ERMES(x,y,z,r,Parameters):
    Parameters.update(CoilDisplacement=[x,y,z],Rotation=r)
    return Namespace(**Parameters)

def GPR_Opt(X,model):
    X = torch.tensor(np.atleast_2d(X),dtype=torch.float32)
    # Function value
    Pred = model(X).mean.detach().numpy()
    # Gradient
    Grad = model.Gradient(X)[0]
    # print(X,Pred)
    return Pred, Grad

def constraint(X, model, DesPower):
    '''
    Constraint which must be met during the optimisation of func. This specifies
    that the power must be greater than or equal to 'DesPower'
    '''
    X = torch.tensor(np.atleast_2d(X),dtype=torch.float32)
    # Function value
    Pred = model(X).mean.detach().numpy()
    constr = Pred - DesPower
    return constr

def dconstraint(X, model, DesPower):
    '''
    Constraint which must be met during the optimisation of func. This specifies
    that the power must be greater than or equal to 'DesPower'
    '''
    X = torch.tensor(np.atleast_2d(X),dtype=torch.float32)
    # Gradient
    Grad = model.Gradient(X)[0]

    return Grad


#
