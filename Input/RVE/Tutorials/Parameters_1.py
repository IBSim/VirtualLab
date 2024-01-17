from types import SimpleNamespace as Namespace
import numpy as N
##########################
######## Meshing #########
##########################

##########################
####### Simulation #######
##########################
#
modelib = Namespace()

modelib.Name = ['microstructure0']


modelib.dislocationline = [1e13]
modelib.dislocationloop = [1e22]
modelib.prec=[1e21]
modelib.b=[.01]
modelib.dim=[15]
modelib.temp=[500]
modelib.strainrate=[1e-11]
