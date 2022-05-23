import os
import sys
sys.dont_write_bytecode=True
from types import SimpleNamespace as Namespace
from importlib import import_module
import copy
import Scripts.Common.VLFunctions as VLF

def Setup(VL, Run=True):
    '''
    This function setups VirtualLab (creative name i know) 
    and so is called first.
    It take in a namespaces VL which is  combination of 
    It takes in a bool value Run to allow you 
    to turn the funcion ON/OFF within VirtualLab.

    For your real package delete this doc string and place 
    a Oneline description of Package here.
    '''

# From here the ultimate aim is to create a nested set of
# dictionarys containing all the input parameters for each run.
#
# To do this first we combine together Parms_Master, Params_Var, 
# and in our case a custom namespace Test using VL.CreateParameters.

    Namespace_Dicts = VL.CreateParameters(VL.Parameters_Master, VL.Parameters_Var,'Test')

# if Run is False or Namespace_Dicts is empty dont don't do anything
# and return instead.
    if not (Run and Namespace_Dicts): return

# Now we define a blank dict and fill it with the following for loop
    VL.RunData = {}
    # Here RunName is the name of each run which will be used as 
    # the keys for the outer most dict. ParaDict is a dict containing
    #  all the parmeters that were defined for that run.
    for RunName, ParaDict in Namespace_Dicts.items():
        # First we convert Paradict back into a namespace
        # This at first glance may seem pointless. However,
        # it allows us to use hasattr to check input parameters 
        # and handle optional parameters faster than 
        # constantly calling dict.get().
        Parameters = Namespace(**ParaDict)
        Parmams_tmp = {}
        # here we can use if hasattr to check for variable and 
        # add them to the dict. this allows us to handle optional
        #  variables as we wish (either giving them default values
        #  or ignore them).
        if hasattr(Parameters,'required_var'):
            Parmams_tmp['required_var'] = Parameters.required_var
        else:
            print("error varible is required")
        
        if hasattr(Parameters,'optional_var'):
            Parmams_tmp['optional_var'] = Parameters.optional_var

        VL.RunData[RunName] = Parameters_tmp.copy()

