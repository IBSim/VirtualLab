
import numpy as np
import matplotlib.pyplot as plt
import bisect

from Scripts.Common.ML import ML, GPR, NN
from Scripts.Common.Optimisation import optimisation

def GPR_compare(VL,DataDict):
    '''
    Function to compare the performance of GPR models with different kernels
    '''
    Parameters = DataDict['Parameters']

    MLModels = Parameters.FixedModels
    TestData = Parameters.TestData

    TestIn, TestOut = ML.VLGetDataML(VL,TestData)
    TestMetric,TrainMetric = {}, {}
    kernels = []
    for model_name in MLModels:
        model_path = "{}/{}".format(VL.ML.output_dir,model_name)
        model = GPR.GetModel(model_path) # load in model
        kernel = model.ModelParameters['kernel']
        kernels.append(kernel)

        # analyse performance on train dtaa
        TrainIn,TrainOut = model.GetTrainData(to_numpy=True) #  get data used to generate model and convert to numpy
        pred = model.Predict(TrainIn)
        power_rmse,var_rmse = ML.RMSE(pred,TrainOut,axis=0) # calculate normalise root mean squared error
        TrainMetric[kernel] = [power_rmse,var_rmse] #  add to dictionary

        # analyse performance on test data
        pred = model.Predict(TestIn)
        power_rmse,var_rmse = ML.RMSE(pred,TestOut,axis=0)
        TestMetric[kernel] = [power_rmse,var_rmse] # add to dictionary

    fig, axes = plt.subplots(1,3,sharey=True,figsize=(15,5))
    x_point = []
    for _i,kernel in enumerate(kernels):
        i = 5*_i
        train_metric = TrainMetric[kernel]
        test_metric = TestMetric[kernel]
        # plot combined score
        axes[0].scatter([i],[np.mean(train_metric)],marker='x',c='k')
        axes[0].scatter([i+1],[np.mean(test_metric)],marker='o',c='k')
        # plot metrics for power prediction
        train_scatter = axes[1].scatter([i],[train_metric[0]],marker='x',c='k')
        test_scatter =  axes[1].scatter([i+1],[test_metric[0]],marker='o',c='k')
        # plot metrics for variation
        axes[2].scatter([i],[train_metric[1]],marker='x',c='k')
        axes[2].scatter([i+1],[test_metric[1]],marker='o',c='k')

        x_point.append(i+0.5)

    plt.setp(axes, xticks=x_point, xticklabels=kernels)
    axes[0].set_title('Average')
    axes[1].set_title('Power')
    axes[2].set_title('Variation')
    axes[1].legend([train_scatter,test_scatter], ['Train','Test'])
    plt.savefig("{}/GPR.png".format(DataDict['CALC_DIR']))
    plt.close()

def MLP_compare(VL,DataDict):
    '''
    Function to compare the performance of MLP models with different architecture
    '''
    Parameters = DataDict['Parameters']

    MLModels = Parameters.MLModels
    TestData = Parameters.TestData

    TestIn, TestOut = ML.VLGetDataML(VL,TestData)
    TestMetric,TrainMetric = {}, {}
    Architectures = []
    for model_name in MLModels:
        model_path = "{}/{}".format(VL.ML.output_dir,model_name)
        model = NN.GetModel(model_path) # load in model
        architecture = model.ModelParameters['Architecture']
        arch_str = '_'.join(map(str,architecture))
        Architectures.append(arch_str)

        # analyse performance on train dtaa
        TrainIn,TrainOut = model.GetTrainData(to_numpy=True) #  get data used to generate model and convert to numpy
        pred = model.Predict(TrainIn)
        power_rmse,var_rmse = ML.RMSE(pred,TrainOut,axis=0) # calculate normalise root mean squared error
        TrainMetric[arch_str] = [power_rmse,var_rmse] #  add to dictionary

        # analyse performance on test data
        pred = model.Predict(TestIn)
        power_rmse,var_rmse = ML.RMSE(pred,TestOut,axis=0)
        TestMetric[arch_str] = [power_rmse,var_rmse] # add to dictionary

    fig, axes = plt.subplots(1,3,sharey=True,figsize=(15,5))
    x_point = []
    for _i,arch_str in enumerate(Architectures):
        i = 5*_i
        train_metric = TrainMetric[arch_str]
        test_metric = TestMetric[arch_str]
        # plot combined score
        axes[0].scatter([i],[np.mean(train_metric)],marker='x',c='k')
        axes[0].scatter([i+1],[np.mean(test_metric)],marker='o',c='k')
        # plot metrics for power prediction
        train_scatter = axes[1].scatter([i],[train_metric[0]],marker='x',c='k')
        test_scatter =  axes[1].scatter([i+1],[test_metric[0]],marker='o',c='k')
        # plot metrics for variation
        axes[2].scatter([i],[train_metric[1]],marker='x',c='k')
        axes[2].scatter([i+1],[test_metric[1]],marker='o',c='k')

        x_point.append(i+0.5)

    plt.setp(axes, xticks=x_point, xticklabels=Architectures)
    axes[0].set_title('Average')
    axes[1].set_title('Power')
    axes[2].set_title('Variation')
    axes[1].legend([train_scatter,test_scatter], ['Train','Test'])
    plt.savefig("{}/MLP.png".format(DataDict['CALC_DIR']))
    plt.close()


def Insight_MLP(VL,DataDict):
    Parameters = DataDict['Parameters']
    MLModel = Parameters.MLModel

    model_path = "{}/{}".format(VL.ML.output_dir,MLModel)
    model = NN.GetModel(model_path) # load in model
    _Insight(DataDict,model)

def Insight_GPR(VL,DataDict):
    Parameters = DataDict['Parameters']
    MLModel = Parameters.MLModel

    model_path = "{}/{}".format(VL.ML.output_dir,MLModel)
    model = GPR.GetModel(model_path) # load in model
    _Insight(DataDict,model)

def _Insight(DataDict,model):
    seed=100

    # use a unit hypercube for better equality of gradients.
    # As a result scale inputs is false as its on the input range model was trained on
    bounds = [[0,1]]*model.Dataspace.NbInput 
    fnc_args = [False,False] # dont scale inputs and dont rescale outputs (everything in [0,1] range)

    # get the minima and maxima for each output & the coordinates to deliver it
    # model.Gradient returns the predicted value and the gradient
    # dont scale outputs during optimisation as it can give bad solutions when values are small
    extrema_inputs, extrema_val = optimisation.GetExtrema(model.Gradient,100,bounds,fnc_args=fnc_args,seed=seed)
    power_min, var_min = extrema_val[0]
    power_min_input, var_min_input = extrema_inputs[0]
    power_max, var_max = extrema_val[1]
    power_max_input, var_max_input = extrema_inputs[1]
    # create an envelope of power versus variation

    Power = np.linspace(power_min,power_max,10)
    VarMin,VarMax = [],[]
    for required_power in Power[1:-1]: # don't need min and max power as these are already calcuated
        # get a dictionary for the constraint
        constraint_dict = optimisation.FixedBound(required_power,model.Gradient,func_args=[*fnc_args,0])
        _cd,_val = optimisation.GetExtrema(model.Gradient,10,bounds,fnc_args=[*fnc_args,1],seed=seed,constraints=constraint_dict)
        _varmin,_varmax = _val
        VarMin.append(_varmin); VarMax.append(_varmax)

    # add the value for variation at the minimum and maximum power configuration
    var_at_Pmin = model.Predict(power_min_input,*fnc_args)[0,1]
    var_at_Pmax = model.Predict(power_max_input,*fnc_args)[0,1]
    VarMin = [var_at_Pmin] + VarMin + [var_at_Pmax]
    VarMax = [var_at_Pmin] + VarMax + [var_at_Pmax]
    # add var minimum point
    PowerMin = Power.tolist()
    power_at_Vmin = model.Predict(var_min_input,*fnc_args)[0,0]
    ix = bisect.bisect_left(PowerMin,power_at_Vmin)
    PowerMin.insert(ix,power_at_Vmin) ; VarMin.insert(ix,var_min)
    # add var max point
    PowerMax = Power.tolist()
    power_at_Vmax = model.Predict(var_max_input,*fnc_args)[0,0]
    ix = bisect.bisect_left(PowerMax,power_at_Vmax)
    PowerMax.insert(ix,power_at_Vmax) ; VarMax.insert(ix,var_max)

    current = 1000 # an example value for the current in the coil
    # Rescale data to correct range
    PowerMin = model.RescaleOutput(np.array(PowerMin),index=0) # only want to rescale the first output (power)
    PowerMax = model.RescaleOutput(np.array(PowerMax),index=0) # only want to rescale the first output (power)
    PowerMin,PowerMax = PowerMin*current**2,PowerMax*current**2
    VarMin = model.RescaleOutput(np.array(VarMin),index=1) # only want to rescale the second output (variation)
    VarMax = model.RescaleOutput(np.array(VarMax),index=1) # only want to rescale the second output (variation)
    VarMin,VarMax = VarMin*current**2,VarMax*current**2
    # additional important data points needed
    data = [[power_min,var_at_Pmin],[power_max,var_at_Pmax],[power_at_Vmin,var_min],[power_at_Vmax,var_max]]
    data = model.RescaleOutput(np.array(data))*current**2

    plt.figure()
    plt.plot(PowerMin,VarMin,label='lower bound',linestyle='--',c='k')
    plt.plot(PowerMax,VarMax,label='upper bound',linestyle='-.',c='k')
    plt.scatter(data[:2,0],data[:2,1],c='k',marker='x',label='power extreme')
    plt.scatter(data[2:,0],data[2:,1],c='k',marker='^',label='variation extreme')
    plt.legend()
    plt.xlabel('Power (W)')
    plt.ylabel('Variation')
    plt.title("Envelope of power versus variation\n for a current of {} A".format(current))
    plt.savefig("{}/Envelope.png".format(DataDict['CALC_DIR']))
    plt.close()  



