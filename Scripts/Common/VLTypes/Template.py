from types import SimpleNamespace as Namespace
import copy

def Setup(VL, Run=True):
    '''
    This function setups VirtualLab to run our custom package
    (creative name i know) and so is called first.

    The ultimate aim is to create a nested set of
    dictionaries containing all the input parameters from the 
    various namespaces for each run. Using this we can then
    call our package from Run.

    It takes in an instance of the VirtualLab class, usually
    this is created with the python self parameter when 
    calling this function within VirtualLab.py.

    In this case it gives us convininet access to params_var, 
    parms_master and the create parameters function. 
    It also gives us a way of sharing the final run paramters
    with the Run function.
    
    The second input parmeter is a bool value Run to allow you 
    to turn the function ON/OFF easily within VirtualLab.

    For your real package you will want to delete this doc 
    string and place a description of your Package here.
    '''

# First we combine together Parms_Master, Params_Var, 
# and in our case a custom namespace 'Example' into one object 
# using VL.CreateParameters.

    Namespace_Dicts = VL.CreateParameters(VL.Parameters_Master, VL.Parameters_Var,'Example')

# If Run is False or Namespace_Dicts is empty dont don't do anything
# and return instead.
    if not (Run and Namespace_Dicts): return

# Now we define a blank dict to contain a dict for each run with 
# its input parameters. Note: The name of this needs to be unique
# to your package. As such it's best to stick with the naming 
# convention of NamespaceData. 
# So for this case we use the 'Example' namespace so we call 
# our Dict ExampleData.
#   
    VL.ExampleData = {}
    # The nested dicts are created and filled with the following for loop.
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
        Params_tmp = {}
        # here we can use if hasattr to check for variables and 
        # add them to the dict. This allows us to handle optional
        #  variables as we wish (either giving them default values
        #  or ignore them).
        if hasattr(Parameters,'required_var'):
            Params_tmp['required_var'] = Parameters.required_var
        else:
            print(f"error required varible 'required_var' not found in Namespace.")
        
        if hasattr(Parameters,'optional_var'):
            Params_tmp['optional_var'] = Parameters.optional_var

        VL.ExampleData[RunName] = Params_tmp.copy()

def Run(VL):
    '''
    This is the function that is called by VirtualLab to actually
    run your code. 

    Much like setup it takes in an instance of the VirtualLab class,
    usually this is created with the python self parameter when 
    calling this function within VirtualLab.py.

    In this case it is used to gives run access to the input
    parmaeters for each run.
    '''
# First we need to import all the packages for our Scripts to run
# In this case I have created a file Example.py in 
# scripts/common/VLPackages that has a function main that simply
# generates a random number between two integers.
#  
# It takes in three parameters min max and an optional list of 
# numbers to avoid it then prints to the screen the random number
# and it's factors.
#
# Note: these need not be in this directory, its just a convinent
# place to put them. The stuff you import can be packages you have 
# installed into the VirtualLab enviroment via pip/conda.

    from Scripts.Common.VLPackages.Example import main

# Next we need to check that we have some input parameters to create 
# our runs.The easiest way to do this is to check if ExampleData 
# exists.
    if not VL.ExmpleData: return
# write output to screen using VL.Logger
    VL.Logger('\n### Starting Example ###\n', Print=True)

# The keys in this dict are the name of each run so here 
# we loop over each run in turn and call main with different
# parmaters.
# 
# Note: in python you can use **dict as an input to a function.
# This passes in each key as a named input to the function.
# In this case each dict has three keys so we have our three inputs
# min, max and avoid.
#  
    for key in VL.ExampleData.keys():
        main(**VL.ExampleData[key])

    VL.Logger('\n### Example Complete ###',Print=True)