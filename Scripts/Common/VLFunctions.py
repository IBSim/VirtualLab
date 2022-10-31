import sys
import os
from types import SimpleNamespace as Namespace
import pickle
from importlib import import_module, reload

import numpy as np

sys.dont_write_bytecode=True

def WriteArgs(path,Args):
    with open(path,'wb') as f:
        pickle.dump(Args,f)

def GetArgs(path):
    with open(path,'rb') as f:
        Args = pickle.load(f)
    return Args

def GetParameterArgs(name='ParameterArgs'):
    ArgDict = {}
    Args=None
    for parsed in sys.argv:
        if parsed.startswith(name):
            path = parsed.split('=')[1]
            Args = GetArgs(path)
    return Args

def GetFunc(FilePath, funcname):
    path,ext = os.path.splitext(FilePath)
    dirname = os.path.dirname(path)
    basename = os.path.basename(path)

    sys.path.insert(0,dirname)
    module = import_module(basename) #reload?
    sys.path.pop(0)

    func = getattr(module, funcname, None)
    return func

def FileFuncSplit(FileInfo,default_func_name):
    if type(FileInfo) in (list,tuple):
        # Split up file name and function name (if given)
        if len(FileInfo)==2:
            file_name,func_name = FileInfo
        else:
            WarningMessage("If file name is not a tuple/list it must have 2 entries: First entry is the file name and the second is the function name.")
            filename, func_name = FileInfo[0], default_func_name
    else:
        # set FuncName to default value
        file_name, func_name = FileInfo, default_func_name

    return file_name, func_name

def ImportUpdate(ParameterFile,ParaDict):
    Parameters = ReadParameters(ParameterFile)
    NewDict = {}
    for Var, Value in Parameters.__dict__.items():
        if Var.startswith('__'): continue
        NewDict[Var] = Value
    for Var, Value in ParaDict.items():
        NewDict[Var] = Value
    return NewDict

def ReadParameters(paramfile):
    paramdir = os.path.dirname(paramfile)
    paramname = os.path.splitext(os.path.basename(paramfile))[0]
    sys.path.insert(0,paramdir)
    try:
        Parameters = reload(import_module(paramname))
    except ImportError:
        parampkl = "{}/.{}.pkl".format(paramdir,paramname)
        with open(parampkl,'rb') as f:
            Parameters = pickle.load(f)
    sys.path.pop(0)
    return Parameters

def ReadData(datapkl):
    DataDict = {}
    with open(datapkl, 'rb') as fr:
        try:
            while True:
                pkldict = pickle.load(fr)
                DataDict = {**pkldict}
        except EOFError:
            pass
    return DataDict

def WriteData(FileName, Data, pkl=True):
    # Check Data type
    if type(Data)==dict:
        DataDict = Data
    elif type(Data)==Namespace:
        DataDict = Data.__dict__
    else:
        print('Unknown type')

    # Write data as readable text
    VarList = []
    for VarName, Val in DataDict.items():
        if type(Val)==str: Val = "'{}'".format(Val)
        VarList.append("{} = {}\n".format(VarName, Val))
    Pathstr = ''.join(VarList)

    with open(FileName,'w+') as f:
        f.write(Pathstr)

    # Create hidden pickle file (ensures importing is possible)
    if pkl:
        dirname = os.path.dirname(FileName)
        basename = os.path.splitext(os.path.basename(FileName))[0]
        pklname = "{}/.{}.pkl".format(dirname,basename)
        try:
            with open(pklname,'wb') as f:
                pickle.dump(Data,f)
        except :
            print('Could not pickle')

def ASCIIname(names):
    namelist = []
    for name in names:
        lis = [0]*80
        lis[:len(name)] = list(map(ord,name))
        namelist.append(lis)
    res = np.array(namelist)
    return res

def WarningMessage(message):
    warning = "\n======== Warning ========\n\n"\
        "{}\n\n"\
        "=========================\n\n".format(message)
    return warning

def ErrorMessage(message):
    error = "\n========= Error =========\n\n"\
        "{}\n\n"\
        "=========================\n\n".format(message)
    return error

def CheckFile(FilePath,Attr=None):
    FileExist = os.path.isfile(FilePath)
    FuncExist = True
    if not FileExist:
        pass
    elif Attr:
        func = GetFunc(FilePath,Attr)
        if func==None: FuncExist = False

    return FileExist, FuncExist

def FileFunc(DirName, FileName, ext = 'py', FuncName = 'Single'):
    if type(FileName) in (list,tuple):
        if len(FileName)==2:
            FileName,FuncName = FileName
        else:
            print('Error: If FileName is a list it must have length 2')
    FilePath = "{}/{}.{}".format(DirName,FileName,ext)

    return FilePath,FuncName

def ImportUpdate(ParameterFile,ParaDict):
    Parameters = ReadParameters(ParameterFile)
    for Var, Value in Parameters.__dict__.items():
        if Var.startswith('__'): continue
        if Var in ParaDict: continue
        ParaDict[Var] = Value

def ReadParameters(paramfile):
    paramdir = os.path.dirname(paramfile)
    paramname = os.path.splitext(os.path.basename(paramfile))[0]
    sys.path.insert(0,paramdir)
    try:
        Parameters = reload(import_module(paramname))
    except ImportError:
        parampkl = "{}/.{}.pkl".format(paramdir,paramname)
        with open(parampkl,'rb') as f:
            Parameters = pickle.load(f)
    sys.path.pop(0)
    return Parameters

def ReadData(datapkl):
    DataDict = {}
    with open(datapkl, 'rb') as fr:
        try:
            while True:
                pkldict = pickle.load(fr)
                DataDict = {**pkldict}
        except EOFError:
            pass
    return DataDict

def WriteData(FileName, Data, pkl=True):
    # Check Data type
    if type(Data)==dict:
        DataDict = Data
    elif type(Data)==Namespace:
        DataDict = Data.__dict__
    else:
        print('Unknown type')

    # Write data as readable text
    VarList = []
    for VarName, Val in DataDict.items():
        if type(Val)==str: Val = "'{}'".format(Val)
        VarList.append("{} = {}\n".format(VarName, Val))
    Pathstr = ''.join(VarList)

    with open(FileName,'w+') as f:
        f.write(Pathstr)

    # Create hidden pickle file (ensures importing is possible)
    if pkl:
        dirname = os.path.dirname(FileName)
        basename = os.path.splitext(os.path.basename(FileName))[0]
        pklname = "{}/.{}.pkl".format(dirname,basename)
        try:
            with open(pklname,'wb') as f:
                pickle.dump(Data,f)
        except :
            print('Could not pickle')

def WarningMessage(message):
    warning = "\n======== Warning ========\n\n"\
        "{}\n\n"\
        "=========================\n\n".format(message)
    return warning

def ErrorMessage(message):
    error = "\n========= Error =========\n\n"\
        "{}\n\n"\
        "=========================\n\n".format(message)
    return error

def VerifyParameters(ParametersNS,vars):
    return list(set(vars) - set(ParametersNS.__dict__))





def MaterialProperty(matarr,Temperature):
    if len(matarr) in (1,2): return matarr[-1]
    else: return np.interp(Temperature, matarr[::2], matarr[1::2])

def Interp_2D(Coordinates,Connectivity,Query):
    Nodes = np.unique(Connectivity.flatten())
    _Ix = np.searchsorted(Nodes,Connectivity)
    a = Coordinates[_Ix]

    a1,a2 = a[:,:,0],a[:,:,1]
    biareas = []
    for ls in [[1,2],[2,0],[0,1]]:
        _a1,_a2 = a1[:,ls], a2[:,ls]
        _d = np.ones((len(_a1),1))
        _a1 = np.concatenate((_a1,_d*Query[0]),axis=1)
        _a2 = np.concatenate((_a2,_d*Query[1]),axis=1)
        _c = np.stack((_a1,_a2,np.ones(_a1.shape)),axis=1)
        _c = np.array(_c,dtype=np.float)
        # print(_c.sum())
        _area = 0.5*np.linalg.det(_c)
        biareas.append(_area)
    biareas = np.array(biareas).T

    sign_area = np.sign(biareas)
    sum_sign = np.abs(sign_area.sum(axis=1))
    elemix = (sum_sign==3).nonzero()[0]
    if len(elemix)==0:
        _sum = (sign_area==0).sum(axis=1)
        for i in range(1,3):
            elemix = ((_sum==i) * (sum_sign==3-i)).nonzero()[0]
            if len(elemix)>0: break

        if len(elemix)==0:
            print('Outside of domain')
            return None
    elemix = elemix[0]

    # get weighting for each contribution
    biarea = biareas[elemix]
    weighting = biarea/biarea.sum()
    nds = Connectivity[elemix,:]

    return nds, weighting



def ParametersVar(arglist):
    return iter(arglist)
