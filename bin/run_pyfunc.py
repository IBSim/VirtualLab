import sys
import os
import pickle

''' 
File to execute python functions as a standalone. This is mostly used to run 
python function within containers.
funcfile (arg1) - path to a python file
funcname (arg2) - name of function in funcfile
argfile (arg3) - path to pickled file which contains args and kwargs to be passed to funcname
'''

funcfile = sys.argv[1]
funcname = sys.argv[2]
argfile = sys.argv[3]

# get directory of funcile to add to sys.path
func_dir = os.path.dirname(funcfile) 
sys.path.insert(0,func_dir)

# name of python file to import 
func_basename = os.path.splitext(os.path.basename(funcfile))[0]
module = __import__(func_basename)

# get function 'funcname' from the imported module
func = getattr(module,funcname)

sys.path.pop(0) # remove function directory from sys.path

# get args and kwargs from argfile
with open(argfile,'rb') as handle:
    args,kwargs = pickle.load(handle)

# execute python function
func(*args,**kwargs) 


