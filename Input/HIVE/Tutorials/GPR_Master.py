from types import SimpleNamespace as Namespace
DA = Namespace()

##########################
#### Machine Learning ####
##########################

DA.Name = 'ML/GPR'

DA.File = 'CoilConfig_GPR'

DA.Device = 'cpu'
DA.NbTorchThread = 2

#========================
#Training
DA.Train = 1 # flag whether to train or not
DA.DataFile = 'Data.hdf5'
DA.Kernel = 'RBF'
DA.Noise = False
DA.lr = 0.01
DA.Iterations = 10000
DA.TrainData = "Halton/PU_3" #training data
DA.TrainNb = 100 # Nb training points to consider

DA.TestData = "TestData/PU_3"
DA.DataSplit = 0.8 # train/test split - decides how much test data to use

#=========================

DA.Input = None

# Optimisation 1: Find the max power
DA.MaxPowerOpt = {'NbInit':20, 'Verify':1, 'NewSim':1}
