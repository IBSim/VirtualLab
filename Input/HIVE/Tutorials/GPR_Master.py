from types import SimpleNamespace as Namespace
DA = Namespace()

##########################
#### Machine Learning ####
##########################

DA.Name = 'ML/GPR_Halton'

DA.File = 'CoilConfig_GPR'

#========================
#Training
# flag whether to train or not
DA.Train = 1
# File where data comes from
DA.DataFile = 'Data.hdf5'
# Training data used
DA.TrainData = "Halton/PU_3"
# Nb training points used
DA.TrainNb = 100
# Test data used
DA.TestData = "TestData/PU_3"
# train/test split - decides how much test data to use
DA.DataSplit = 0.8
#Max. number of epoch iterations.
DA.Iterations = 10000
# Parameters for GPR model
DA.Kernel = 'RBF'
DA.Noise = False
DA.lr = 0.01

#=========================

# Optimisation 1: Find the max power
# Verify - runs a simulation to compare againt model prediction
DA.MaxPowerOpt = {'NbInit':20, 'Verify':True}
