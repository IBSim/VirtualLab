import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from types import SimpleNamespace as Namespace
import torch
import gpytorch
import time

from Functions import Uniformity3 as UniformityScore
import ML
from Optimise import FuncOpt
from Sim.PreHIVE import ERMES

dtype = 'float64' # float64 is more accurate for optimisation purposes
torch_dtype = getattr(torch,dtype)

def InputParameters():
    pass

def Single(VL, DADict):

    torch.set_default_dtype(torch_dtype)

    Parameters = DADict["Parameters"]

    NbTorchThread = getattr(Parameters,'NbTorchThread',1)
    torch.set_num_threads(NbTorchThread)
    torch.manual_seed(getattr(Parameters,'Seed',100))

    # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')

    # File where all data is stored
    DataFile = "{}/{}".format(VL.PROJECT_DIR,Parameters.DataFile)

    if Parameters.Train:
        DataSplit = getattr(Parameters,'DataSplit',0.7)
        TrainNb = getattr(Parameters,'TrainNb',None)

        #=======================================================================
        # Get Training & Testing data from file
        MLData = h5py.File(DataFile,'r')

        if Parameters.TrainData in MLData:
            _TrainData = MLData[Parameters.TrainData][:]
        elif "{}/{}".format(VL.StudyName,Parameters.TrainData) in MLData:
            _TrainData = MLData["{}/{}".format(VL.StudyName,Parameters.TrainData)][:]
        else : sys.exit("Training data not found")

        if not TrainNb:
            TrainNb = int(np.ceil(_TrainData.shape[0]*DataSplit))
        TrainData = _TrainData[:TrainNb,:] # Data used for training

        np.save("{}/TrainData".format(DADict["CALC_DIR"]),TrainData)

        TestNb = int(np.ceil(TrainNb*(1-DataSplit)/DataSplit))
        if hasattr(Parameters,'TestData'):
            if Parameters.TestData == Parameters.TrainData:
                TestData = _TrainData[TrainNb:,:]
            elif Parameters.TestData in MLData :
                TestData = MLData[Parameters.TestData][:]
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
    TrainData = TrainData.astype(dtype)
    TestData = TestData.astype(dtype)
    Train_x, Train_y = TrainData[:,:4], TrainData[:,4:]
    Test_x, Test_y = TestData[:,:4], TestData[:,4:]

    # Scale test & train input data to [0,1] (based on training data)
    InputScaler = np.array([Train_x.min(axis=0),Train_x.max(axis=0) - Train_x.min(axis=0)])
    OutputScaler = np.array([Train_y.min(axis=0),Train_y.max(axis=0) - Train_y.min(axis=0)])
    # InputScaler = np.array([np.mean(Train_x,axis=0),np.std(Train_x,axis=0)])
    # OutputScaler = np.array([np.mean(Train_y,axis=0),np.std(Train_y,axis=0)])

    Train_x_scale = ML.DataScale(Train_x,*InputScaler)
    Train_y_scale = ML.DataScale(Train_y,*OutputScaler)
    Train_x_tf = torch.from_numpy(Train_x_scale)
    Train_y_tf = torch.from_numpy(Train_y_scale)

    Test_x_scale = ML.DataScale(Test_x,*InputScaler)
    Test_y_scale = ML.DataScale(Test_y,*OutputScaler)
    Test_x_tf = torch.from_numpy(Test_x_scale)
    Test_y_tf = torch.from_numpy(Test_y_scale)

    #======================================================================

    Train_P_tf,Train_V_tf = Train_y_tf[:,0],Train_y_tf[:,1]
    Test_P_tf,Test_V_tf = Test_y_tf[:,0],Test_y_tf[:,1]

    if getattr(Parameters,'Noise',True):
        PowerLH = gpytorch.likelihoods.GaussianLikelihood()
        VarLH = gpytorch.likelihoods.GaussianLikelihood()
    else:
        # Noise can't be zero in gpytorch so set it to very small value
        sig = 0.00001*torch.ones(Train_x_tf.shape[0])
        PowerLH = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(sig)
        VarLH = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(sig)


    Power = ML.ExactGPmodel(Train_x_tf, Train_P_tf, PowerLH,
                    Parameters.Kernel)
    Variation = ML.ExactGPmodel(Train_x_tf, Train_V_tf, VarLH,
                    Parameters.Kernel)

    if Parameters.Train:
        # Power
        lr = getattr(Parameters,'lr', 0.01)
        DADict['Data']['MSE'] = MSEvals = {}
        print('Training Power')
        Conv_P,MSE_P = Power.Training(PowerLH,Parameters.Iterations,lr=lr,Print=50)
        print()
        ModelFile = '{}/Power.pth'.format(DADict["CALC_DIR"]) # File model will be saved to/loaded from
        torch.save(Power.state_dict(), ModelFile)

        if MSE_P:
            plt.figure()
            plt.plot(np.array(MSE_P)*OutputScaler[1,0]**2)
            plt.savefig("{}/MSE_Power.png".format(DADict["CALC_DIR"]))
            plt.close()

        with torch.no_grad(), gpytorch.settings.max_cholesky_size(1500), gpytorch.settings.debug(False):
            Power_test = ML.MSE(Power(Test_x_tf).mean.numpy(), Test_y_scale[:,0])
            Power_train = ML.MSE(Power(Train_x_tf).mean.numpy(), Train_y_scale[:,0])
        Power_test *= OutputScaler[1,0]**2
        Power_train *= OutputScaler[1,0]**2
        print("Train MSE: {}\nTest MSE: {}\n".format(Power_train,Power_test))

        print(Power_test,Power_train)
        MSEvals["Power_Test"] = Power_test
        MSEvals["Power_Train"] = Power_train

        # Variation
        print('Training Variation')
        Conv_V,MSE_V = Variation.Training(VarLH,Parameters.Iterations,lr=lr,Print=100)
        print()
        ModelFile = '{}/Variation.pth'.format(DADict["CALC_DIR"]) # File model will be saved to/loaded from
        torch.save(Variation.state_dict(), ModelFile)

        if MSE_V:
            plt.figure()
            plt.plot(MSE_V)
            plt.savefig("{}/MSE_Variation.png".format(DADict["CALC_DIR"]))
            plt.close()

        with torch.no_grad(), gpytorch.settings.max_cholesky_size(1500), gpytorch.settings.debug(False):
            Vari_test = ML.MSE(Variation(Test_x_tf).mean.numpy(), Test_y_scale[:,1])
            Vari_train = ML.MSE(Variation(Train_x_tf).mean.numpy(), Train_y_scale[:,1])
        Vari_test  *= OutputScaler[1,1]**2
        Vari_train *= OutputScaler[1,1]**2
        print("Train MSE: {}\nTest MSE: {}\n".format(Vari_train,Vari_test))
        MSEvals["Variation_Test"] = Vari_test
        MSEvals["Variation_Train"] = Vari_train

        with torch.no_grad(), gpytorch.settings.max_cholesky_size(1500), gpytorch.settings.debug(False):
            Test_MSE_P = ML.MSE(Power(Test_x_tf).mean.numpy(), Test_y_scale[:,0])
            Test_MSE_P *= OutputScaler[1,0]**2 #scale MSE to correct range

            Train_MSE_P = ML.MSE(Power(Train_x_tf).mean.numpy(), Train_y_scale[:,0])
            Train_MSE_P *= OutputScaler[1,0]**2 #scale MSE to correct range

        fnt=36
        plt.figure(figsize=(15,10))
        plt.plot(Conv_P,label='Power',c=plt.cm.gray(0))
        plt.plot(Conv_V, label='Variation',c=plt.cm.gray(0.5))

        plt.ylabel("Loss (MLL)",fontsize=fnt)
        plt.xlabel("Epochs",fontsize=fnt)
        plt.legend(fontsize=fnt)
        plt.savefig("{}/Convergence.eps".format(DADict["CALC_DIR"]),dpi=600)
        plt.close()

    else:
        # Power
        state_dict_P = torch.load('{}/Power.pth'.format(DADict["CALC_DIR"]))
        Power.load_state_dict(state_dict_P)
        PowerLH.eval(); Power.eval()

        Variation
        state_dict_V = torch.load('{}/Variation.pth'.format(DADict["CALC_DIR"]))
        Variation.load_state_dict(state_dict_V)
        Variation.eval(); VarLH.eval()

    if getattr(Parameters,'Input',None) != None:
        with torch.no_grad():
            x_scale = ML.DataScale(np.atleast_2d(Parameters.Input),*InputScaler)
            x_scale = torch.tensor(x_scale, dtype=torch_dtype)
            y = Power(x_scale)
            y_mean = ML.DataRescale(y.mean.numpy(),*OutputScaler[:,0])
            y_stddev = ML.DataRescale(y.stddev,0,OutputScaler[1,0])
            print(y_mean)
            print(y_stddev)
            # print(y_mean)
            DADict['Output'] = y_mean.tolist()
            DADict['Output_Var'] = y_stddev.tolist()



    if 0:
        model=Power
        Slice = 'xy'
        Res = 'Power' # 'Uniformity'
        Component = 'mag'
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
        # print(grid)

        grid_tf = torch.tensor(grid, dtype=torch_dtype)
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

        imres = ML.DataRescale(imres,*OutputScaler[:,0])

        # min and max value for global colour bar
        IMMin, IMMax = imres.min(axis=0), imres.max(axis=0)

        imres = imres.reshape(_disc+[*imres.shape[1:]])
        # scaled vales for major axis
        arr1 = ML.DataRescale(DiscMaj,*InputScaler[:,AxMaj[0]])
        arr2 = ML.DataRescale(DiscMaj,*InputScaler[:,AxMaj[1]])

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
        fig.text(0.5, 0.04, InTag[AxMaj[1]].capitalize(), va='center')
        # plt.show()
        plt.savefig("{}/4D_Plot.png".format(DADict["CALC_DIR"]))
        plt.close()



    # Get bounds of data for optimisation
    bnds = list(zip(Train_x_scale.min(axis=0), Train_x_scale.max(axis=0)))

    MeshFile = "{}/AMAZE_Sample.med".format(VL.MESH_DIR)
    ERMES_Parameters = {'CoilType':'HIVE',
                        'Current':1000,
                        'Frequency':1e4,
                        'Materials':{'Block':'Copper_NL', 'Pipe':'Copper_NL', 'Tile':'Tungsten_NL'}}

    OutString = ""

    # Optimsation 1: Find the point of max power
    # Find the point(s) which give the maximum power
    print("Locating optimum configuration(s) for maximum power")
    NbInit = Parameters.MaxPowerOpt.get('NbInit',20)
    np.random.seed(123)
    init_guess = np.random.uniform(0,1,size=(NbInit,4))

    if True:
        Optima = FuncOpt(GPR_Opt, init_guess, find='max', tol=0.01,
                         order='decreasing',
                         bounds=bnds, jac=True, args=[Power])
    else:
        ''' Test speed difference for optimisation (no constraints)'''
        m = 5
        tots,totp = 0, 0
        for _ in range(m):
            st = time.time()
            Optima = FuncOpt(GPR_Opt, init_guess, find='max', tol=0.01,
                            order='decreasing', version='individual',
                            bounds=bnds, jac=True, args=[Power])
            ends = time.time()-st
            tots+=ends

            st = time.time()
            Optima = FuncOpt(GPR_Opt, init_guess, find='max', tol=0.01,
                             order='decreasing',
                             bounds=bnds, jac=True, args=[Power])

            endp = time.time()-st
            totp+=endp

        print('Scipy',tots/m)
        print('Parallel',totp/m)

    MaxPower_cd,MaxPower_val = Optima

    with torch.no_grad():
        MaxPower_cd_tf = torch.tensor(MaxPower_cd, dtype=torch_dtype)
        MPPower = Power(MaxPower_cd_tf)
        MPVar = Variation(MaxPower_cd_tf)
    MaxPower_cd = ML.DataRescale(MaxPower_cd, *InputScaler)
    MPMean = np.vstack((MPPower.mean.numpy(),MPVar.mean.numpy())).T
    MPStd = np.vstack((MPPower.stddev.detach(),MPVar.stddev.detach())).T

    MaxPower_val = ML.DataRescale(MPMean,*OutputScaler)

    MaxPower_std = ML.DataRescale(MPStd,0,OutputScaler[1,:])
    print("    {:7}{:7}{:7}{:11}{:9}".format('x','y','z','r','Power'))
    for coord, val, std in zip(MaxPower_cd,MaxPower_val,MaxPower_std):
        print("({0:.4f},{1:.4f},{2:.4f},{3:.4f}) ---> {4:.2f} W ({6}) {5:.2f} ({7})".format(*coord,*val,*std))
    print()

    DADict["Data"]['MaxPower'] = MaxPower = {'x':MaxPower_cd[0],'y':MaxPower_val[0,0]}

    if Parameters.MaxPowerOpt.get('Verify',True):
        print("Checking results at optima\n")
        ERMESMaxPower = '{}/MaxPower.rmed'.format(DADict["CALC_DIR"])
        EMParameters = Param_ERMES(*MaxPower_cd[0],ERMES_Parameters)
        RunERMES = Parameters.MaxPowerOpt.get('NewSim',True)

        JH_Vol, Volumes, Elements, JH_Node = ERMES(VL,MeshFile,ERMESMaxPower,
                                                EMParameters, DADict["TMP_CALC_DIR"],
                                                RunERMES, GUI=0)
        Watts = JH_Vol*Volumes
        # Power & Uniformity
        Power = np.sum(Watts)
        JH_Node /= 1000**2
        Uniformity = UniformityScore(JH_Node,ERMESMaxPower)
        # print(MaxPower_val)
        print("Anticipated power at optimum configuration: {0:.2f} W".format(MaxPower_val[0,0]))
        print("Actual power at optimum configuration: {:.2f} W\n".format(Power))

        MPstr = "Anticipated power at optimum configuration: {:.2f} W\n"\
        "Actual power at optimum configuration: {:.2f} W\n\n".format(MaxPower_val[0,0],Power)

        OutString+=MPstr

        MaxPower["target"] = Power


    if hasattr(Parameters,'DesPowerOpt'):
        if Parameters.DesPowerOpt['Power'] >= MaxPower_val[0,0]:
            print('DesiredPower greater than power available.\n')
        else:
            print("Locating optimum configuration(s) for maximum uniformity, ensuring power >= {} W".format(Parameters.DesPowerOpt['Power']))
            NbInit = Parameters.DesPowerOpt.get('NbInit',20)
            np.random.seed(123)
            init_guess = np.random.uniform(0,1,size=(NbInit,4))
            # constraint to ensure power requirement is met
            DesPower_norm = ML.DataScale(Parameters.DesPowerOpt['Power'], *OutputScaler[:,0]) #scale power
            con1 = {'type': 'ineq', 'fun': constraint, 'jac':dconstraint, 'args':(Power, DesPower_norm)}

            if True:
                Optima = FuncOpt(GPR_Opt, init_guess, find='min', tol=0.01,
                                order='increasing',
                                bounds=bnds, jac=True,
                                args=[Variation], constraints=con1)
            else:
                m=5
                tots,totp = 0,0
                for _ in range(m):
                    ''' Test speed difference for optimisation with no constraint'''
                    st = time.time()
                    Optima = FuncOpt(GPR_Opt, init_guess, find='min', tol=0.01,
                                     order='increasing', version='individual',
                                     bounds = bnds, jac=True,
                                     args=[Variation], constraints=con1)
                    ends = time.time() - st
                    tots+=ends
                    print(ends)
                    st = time.time()
                    Optima = FuncOpt(GPR_Opt, init_guess, find='min', tol=0.01,
                                    order='increasing',
                                    bounds=bnds, jac=True,
                                    args=[Variation], constraints=con1)
                    endp = time.time() - st
                    totp+=endp
                    print(endp)
                print('Scipy',tots/m)
                print('Multi',totp/m)


            OptVar_cd,OptVar_val = Optima



            with torch.no_grad():
                OptVar_tf = torch.tensor(OptVar_cd, dtype=torch_dtype)
                _P,_V = Power(OptVar_tf), Variation(OptVar_tf)
                PV = _P.mean.numpy(),_V.mean.numpy()

            OptVar_val = ML.DataRescale(np.array(PV).T,*OutputScaler)
            OptVar_cd = ML.DataRescale(OptVar_cd, *InputScaler)

            print("    {:7}{:7}{:7}{:11}{:9}{:8}".format('x','y','z','r','Power','Variation'))
            for coord, val in zip(OptVar_cd, OptVar_val):
                print("({:.4f},{:.4f},{:.4f},{:.4f}) ---> {:.2f} W, {:.3f}".format(*coord, *val))
            print()

            DADict["Data"]['C_Var'] = C_Var = {'x':OptVar_cd,'y':OptVar_val}

            if Parameters.DesPowerOpt.get('Verify',True):
                print("Checking results at optima\n")
                ERMESRes = '{}/MinVar_{}.rmed'.format(DADict["CALC_DIR"],Parameters.DesPowerOpt['Power'])
                RunERMES = Parameters.DesPowerOpt.get('NewSim',True)
                EMParameters = Param_ERMES(*OptVar_cd[0],ERMES_Parameters)

                JH_Vol, Volumes, Elements, JH_Node = ERMES(VL,MeshFile,ERMESRes,
                                                    EMParameters, DADict["TMP_CALC_DIR"],
                                                    RunERMES, GUI=0)

                Watts = JH_Vol*Volumes
                # Power & Uniformity
                Power = np.sum(Watts)

                Uniformity = UniformityScore(JH_Node,ERMESRes)

                print("Anticipated at optimum configuration:\nPower: {:.2f} W\n"\
                      "Uniformity: {:.2f}".format(*OptVar_val[0]))
                print("Actual values at optimum configuration:\nPower: {:.2f} W\n"\
                      "Uniformity: {:.2f}".format(Power,Uniformity))

                COstr = "Anticipated at optimum configuration:\nPower: {:.2f} W\n"\
                      "Uniformity: {:.2f}\nActual values at optimum "\
                      "configuration:\nPower: {:.2f} W\nUniformity: {:.2f}"\
                      .format(*OptVar_val[0],Power,Uniformity)

                OutString+=COstr

                C_Var['Target'] = [Power,Uniformity]

    if OutString:
        with open("{}/Summary.txt".format(DADict['CALC_DIR']),'w') as f:
            f.write(OutString)


def Param_ERMES(x,y,z,r,Parameters):
    Parameters.update(CoilDisplacement=[x,y,z],Rotation=r)
    return Namespace(**Parameters)

def GPR_Opt(X,model):
    X = torch.tensor(np.atleast_2d(X),dtype=torch_dtype)
    # Function value
    Pred = model(X).mean.detach().numpy()
    # Gradient
    Grad = model.Gradient(X)
    # print(Grad.shape)
    # print(X,Pred)
    return Pred, Grad

def constraint(X, model, DesPower):
    '''
    Constraint which must be met during the optimisation of func. This specifies
    that the power must be greater than or equal to 'DesPower'
    '''
    X = torch.tensor(np.atleast_2d(X),dtype=torch_dtype)
    # Function value
    Pred = model(X).mean.detach().numpy()
    constr = Pred - DesPower
    return constr

def dconstraint(X, model, DesPower):
    '''
    Constraint which must be met during the optimisation of func. This specifies
    that the power must be greater than or equal to 'DesPower'
    '''
    X = torch.tensor(np.atleast_2d(X),dtype=torch_dtype)
    # Gradient
    Grad = model.Gradient(X)

    return Grad


#
