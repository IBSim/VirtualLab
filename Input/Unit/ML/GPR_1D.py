from types import SimpleNamespace as Namespace

DA = Namespace()

DA.Name = 'Models/GPR_1D'
DA.File = ('ML_GPR','Example_1D')

DA.Limits = [0,10] # limits of function domain
DA.NbTrain = 5 # number ofpoints used for traing the model
DA.NbTest = 5 # number of points used to test the models generalisation

# Model parameters
# kernel: encodes assumptions about the data. The most important parameter
# min_noise: The lower limit for the noise (default is 1e-4 is gpytroch)
# 'noise_init': initial value for noise. Helps avoid bad optima with large noise
DA.ModelParameters = {'kernel':'RBF','min_noise':0,'noise_init':1e-5}

# Training parameters
# Epochs: number of times to loop over the training data
# lr: learning rate
# print: how often to print an update of the training
DA.TrainingParameters = {'Epochs':100,'lr':0.1,'Print':10}
