
import os
import sys

import numpy as np

import VLFunctions as VLF
from Scripts.Common.tools import MEDtools
from Scripts.Common.ML import ML

# ==============================================================================
# Functions for gathering necessary data and writing to file
def CompileData(VL,DADict):
    Parameters = DADict["Parameters"]

    # Top level directory containing directories of simulation results
    ResDir_TLD = "{}/{}".format(VL.PROJECT_DIR,Parameters.CompileData)
    # File which will store extracted data
    DataFile_path = "{}/{}".format(VL.PROJECT_DIR,Parameters.DataFile)

    names, functions, args, kwargs = [],[],[],[]
    for _dict in Parameters.Collect:
        # checks
        if 'Name' not in _dict:
            print(VLF.ErrorMessage("'Name' must be specified in the Collect dictionary."))
            sys.exit()
        else:
            name = _dict['Name']

        if 'Function' not in _dict:
            print(VLF.ErrorMessage("'Function' must be specified in the Collect dictionary."))
            sys.exit()
        else:
            function = _dict['Function']

        # get function
        # TODO: generalise this bit
        fn = globals()[function]

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
        data_path = "{}/{}".format(Parameters.CompileData,name)
        ML.Writehdf(DataFile_path, data_path, data)

def Example1(ResDir_path):
    # Do something with results in ResDir_path
    ans=2
    return ans

def Example2(ResDir_path,a,var=3):
    # Do something with results in ResDir_path
    ans= 2*a + var
    return ans

def Inputs(ResDir_path, InputVariables, Parameters_basename ='Parameters.py'):
    ''' Get values for the variables specified in InputVariables.'''

    paramfile = "{}/{}".format(ResDir_path,Parameters_basename)
    Parameters = VLF.ReadParameters(paramfile)
    Values = ML.GetInputs(Parameters, InputVariables)
    return Values

# ==============================================================================
# Code Aster
def NodalResult(ResDir_path, ResFileName, ResName, GroupName=None):
    ''' Get result 'ResName' at all nodes. Results for certain groups can be
        returned using GroupName argument.'''

    ResFilePath = "{}/{}".format(ResDir_path,ResFileName)
    Temps = MEDtools.NodalResult(ResFilePath,ResName,GroupName=GroupName)

    return  Temps

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

def MaxVMis(ResDir_path, ResFileName, ResName='Temperature'):
    VMis_all = VMisField(ResDir_path, ResFileName, ResName=ResName)
    return VMis_all.max()

def MinVMis(ResDir_path, ResFileName, ResName='Temperature'):
    VMis_all = VMisField(ResDir_path, ResFileName, ResName=ResName)
    return VMis_all.min()

def Power_ERMES(ResDir_path, ResFileName, ResName='Joule_heating', GroupName=None):
    ''' Get result 'ResName' at all nodes. Results for certain groups can be
        returned using GroupName argument.'''


    JH_Node = NodalResult(ResDir_path, ResFileName, ResName, GroupName=GroupName)

    meshdata = MEDtools.MeshInfo("{}/{}".format(ResDir_path,ResFileName))

    NodeIDs = list(range(1,meshdata.NbNodes+1))
    Coor = meshdata.GetNodeXYZ(NodeIDs)
    Sample = meshdata.GroupInfo('Sample')
    Connect = Sample.Connect

    _Ix = np.searchsorted(NodeIDs,Connect)

    # work out volume of each element
    elem_cd = Coor[_Ix]
    v1,v2 = elem_cd[:,1] - elem_cd[:,0], elem_cd[:,2] - elem_cd[:,0]
    v3 = elem_cd[:,3] - elem_cd[:,0]
    cross = np.cross(v1,v2)
    dot = (cross*v3).sum(axis=1)
    Volumes = 1/float(6)*np.abs(dot)

    # work out average joule heating per volume
    _jh = JH_Node[_Ix]
    JH_vol = _jh.mean(axis=1)

    # Calculate power
    Power = (Volumes*JH_vol).sum()

    return Power
