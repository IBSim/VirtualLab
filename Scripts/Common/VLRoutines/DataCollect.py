
import os
import sys

import numpy as np

import VLFunctions as VLF
from Scripts.Common.tools import MEDtools
from Scripts.Common.ML import ML

default_functions = ['Inputs','NodalMED']

# ==============================================================================
# Functions for gathering necessary data and writing to file

def CompileData(VL,DataDict):
    return CompileDataAdd(VL,DataDict,{})

def CompileDataAdd(VL,DataDict,add_funcs):
    Parameters = DataDict["Parameters"]
    Collect = Parameters.Collect
    CompileData = Parameters.CompileData
    group = getattr(Parameters,'Group','')

    # Top level directory containing directories of simulation results
    ResDir_TLD = "{}/{}".format(VL.PROJECT_DIR,CompileData)
    # TODO: check this directory exists

    # File which will store extracted data
    DataFile_path = "{}/{}".format(VL.PROJECT_DIR,Parameters.DataFile)
    Collector(ResDir_TLD,Collect,DataFile_path,add_funcs=add_funcs,group=group)

def Collector(ResDir_TLD, DataCollect, DataFile_path, add_funcs={},  group=None):
    '''
    ResDir_TLD: The results directory which will be iterated over
    DataCollect: A list of dictionaries describing the different information to extract
    DataFile_path: path to the file where data will be stored
    add_funcs: a dictionary of any additional functions required
        '''
    available_funcs = _default_funcs()
    available_funcs.update(add_funcs)

    names, functions, args, kwargs = [],[],[],[]
    for _dict in DataCollect:
        # checks
        if 'Name' not in _dict:
            print(VLF.ErrorMessage("'Name' must be specified in the Collect dictionary."))
            sys.exit()

        if 'Function' not in _dict:
            print(VLF.ErrorMessage("'Function' must be specified in the Collect dictionary."))
            sys.exit()

        if group is not None: name = "{}/{}".format(group,_dict['Name'])
        else: name = _dict['Name']
        func_name = _dict['Function']

        # get function
        if func_name not in available_funcs:
            print(VLF.ErrorMessage("Function '{}' is not available. Please check it has been passed using the func_dict key word argument".format(func_name)))
            sys.exit()       

        fn = available_funcs[func_name]

        names.append(name)
        functions.append(fn)
        args.append(_dict.get('args',()))
        kwargs.append(_dict.get('kwargs',{}))

    # ==========================================================================
    # Go through results in ResDir and extract data using functions, args and kwargs
    data_list = ML.ExtractData_Dir(ResDir_TLD,functions,args,kwargs)

    # ==========================================================================
    # Write the collected data to file
    for name, data in zip(names,data_list):
        ML.Writehdf(DataFile_path, name, data)    


def _default_funcs():
    func_dict = {name:globals()[name] for name in default_functions}
    return func_dict

# ==============================================================================
# useful functions which are likely used repeatedly

def Inputs(ResDir_path, InputVariables, Parameters_basename ='Parameters.py'):
    ''' Get values for the variables specified in InputVariables.'''

    paramfile = "{}/{}".format(ResDir_path,Parameters_basename)
    Parameters = VLF.ReadParameters(paramfile)
    Values = ML.GetInputs(Parameters, InputVariables)
    return Values

def NodalMED(ResDir_path, ResFileName, *args,**kwargs):
    ''' Get result 'ResName' at all nodes. Results for certain groups can be
        returned using GroupName argument.'''
    ResFilePath = "{}/{}".format(ResDir_path,ResFileName)
    return MEDtools.NodalResult(ResFilePath,*args,**kwargs)



def MaxNode(ResDir_path, ResFileName, ResName='Temperature'):
    TempField = NodalField(ResDir_path, ResFileName, ResName=ResName)
    return TempField.max()

def MinNode(ResDir_path, ResFileName, ResName='Temperature'):
    TempField = NodalField(ResDir_path, ResFileName, ResName=ResName)
    return TempField.min()

def VMisField(ResDir_path, ResFileName, ResName='Stress'):
    ''' Get temperature values at all nodes'''

    # Get temperature values from results
    ResFilePath = "{}/{}".format(ResDir_path,ResFileName)
    Stress = MEDtools.ElementResult(ResFilePath,ResName)
    Stress = Stress.reshape((int(Stress.size/6),6))

    VMis = (((Stress[:,0] - Stress[:,1])**2 + (Stress[:,1] - Stress[:,2])**2 + \
              (Stress[:,2] - Stress[:,0])**2 + 6*(Stress[:,3:]**2).sum(axis=1)  )/2)**0.5

    mesh = MEDtools.MeshInfo(ResFilePath)
    cnct = mesh.ConnectByType('Volume')

    # extrapolate element value to node to reduce storage requirements
    sumvmis,sumcount = np.zeros(mesh.NbNodes),np.zeros(mesh.NbNodes)
    for i,vm in zip(cnct,VMis):
        sumvmis[i-1]+=vm
        sumcount[i-1]+=1
    VMis_nd = sumvmis/sumcount

    return VMis_nd


