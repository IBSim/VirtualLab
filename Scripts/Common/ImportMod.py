import sys
import os
from SalomeFunc import GetArgs
from importlib import import_module

ArgDict = GetArgs(sys.argv[1:])
Module = ArgDict.pop('Module')
Function = ArgDict.pop('Function')
ModDir = os.path.dirname(Module)
ModName = os.path.splitext(os.path.basename(Module))[0]

sys.path.insert(0,ModDir)
Mod = import_module(ModName)
Function = getattr(Mod,Function)
Function()
# if ArgDict: Function(ArgDict)
# else : Function()
