import sys
import os
from types import SimpleNamespace as Namespace
import pickle
from importlib import import_module, reload

import h5py
import numpy as np

sys.dont_write_bytecode=True

def GetFunc(FilePath, funcname):
    path,ext = os.path.splitext(FilePath)
    dirname = os.path.dirname(path)
    basename = os.path.basename(path)

    sys.path.insert(0,dirname)
    module = import_module(basename) #reload?
    sys.path.pop(0)

    func = getattr(module, funcname, None)
    return func

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

def VerifyParameters(ParametersNS,vars):
    return list(set(vars) - set(ParametersNS.__dict__))

def MaterialProperty(matarr,Temperature):
    if len(matarr) in (1,2): return matarr[-1]
    else: return np.interp(Temperature, matarr[::2], matarr[1::2])

class MeshInfo():
    def __init__(self, meshfile, meshname = None):
        self.g = h5py.File(meshfile, 'r')
        # Fin the name of the mesh(es) in the file
        names = self.g['ENS_MAA'].keys()

        # If only one mesh in file then set meshname to this
        if len(names) == 1:
            meshname = list(names)[0]
        # If meshname provided check it is in the file, if not error
        elif meshname and str(meshname) not in names:
            err = 'meshname provided not in file'
        # If multiple meshes in the file and no meshname provided, call error
        else :
            err = 'Multiple meshes in file and no meshname given'

        self.MeshName = meshname

        # CnctPath is the path to nodal and element data
        self.CnctPath = "ENS_MAA/{}/-0000000000000000001-0000000000000000001".format(self.MeshName)
        self.NbNodes = self.g["{}/NOE/COO".format(self.CnctPath)].attrs['NBR']

        self.NbElements, self.NbVolumes, self.NbSurfaces, self.NbEdges = 0, 0, 0, 0
        for ElType in self.g["{}/MAI".format(self.CnctPath)].keys():
            value = self.g["{}/MAI/{}/NUM".format(self.CnctPath, ElType)].attrs['NBR']
            if ElType == 'TE4':
                self.NbVolumes = value
            elif ElType == 'TR3':
                self.NbSurfaces = value
            elif ElType == 'SE2':
                self.NbEdges = value
            self.NbElements += value

    def ConnectByType(self,ElemType):
        if ElemType in ['Volume']: ElemType = 'TE4'

        dset = self.g["{}/MAI/{}/NOD".format(self.CnctPath, ElemType)]
        Connectivity = np.reshape(dset[:], (dset.attrs['NBR'],int(dset.shape[0]/dset.attrs['NBR'])), order='F')
        return Connectivity

    def ElementsByType(self, ElemType):
        if ElemType in ['Volume']: ElemType = 'TE4'

        ElemNum = self.g["{}/MAI/{}/NUM".format(self.CnctPath, ElemType)][:]
        return ElemNum

    def __GroupSort__(self):
        grpInfo = {}
        for ElType in self.g["{}/MAI".format(self.CnctPath)].keys():
            for val in np.unique(self.g["{}/MAI/{}/FAM".format(self.CnctPath, ElType)][:]):
                grpInfo[val] = ElType

        Groups = {}
        # Element groups
        ElGrpPath = "FAS/{}/ELEME".format(self.MeshName)
        if ElGrpPath in self.g:
            for Grp in self.g[ElGrpPath].keys():
                # Get unique number associated to elements for grouping
                num = self.g["{}/{}".format(ElGrpPath, Grp)].attrs['NUM']

                grptype = grpInfo[num]
                if grptype not in Groups.keys():
                    Groups[grptype] = {}

                # Find the group(s) associated with this unique number
                for uniname in self.g["{}/{}/GRO/NOM".format(ElGrpPath, Grp)][:]:
                    # Convert name from unicode chars to ascii string
                    charlist = list(map(chr,uniname))
                    asciiname = ''.join(charlist).rstrip(charlist[-1])

                    if asciiname not in Groups[grptype].keys():
                        Groups[grptype][asciiname] = []

                    Groups[grptype][asciiname].append(num)

        # Node groups
        Groups['NODE'] = {}
        NdGrpPath = "FAS/{}/NOEUD".format(self.MeshName)
        if NdGrpPath in self.g:
            for Grp in self.g[NdGrpPath].keys():
                # Get unique number associated to elements for grouping
                num = self.g["{}/{}".format(NdGrpPath, Grp)].attrs['NUM']
                # Find the group(s) associated with this unique number
                for uniname in self.g["{}/{}/GRO/NOM".format(NdGrpPath, Grp)][:]:
                    # Convert name from unicode chars to ascii string
                    charlist = list(map(chr,uniname))
                    asciiname = ''.join(charlist).rstrip(charlist[-1])

                    if asciiname not in Groups['NODE'].keys():
                        Groups['NODE'][asciiname] = []

                    Groups['NODE'][asciiname].append(num)

        self.__GroupInfo__ = Groups

    def GroupTypes(self):
        # This function return a dict of what groups are in the mesh by their type
        if not hasattr(self,'__GroupInfo__'):
            self.__GroupSort__()
        GrpDict = dict.fromkeys(self.__GroupInfo__)
        for key, item in self.__GroupInfo__.items():
            GrpDict[key] = list(item.keys())
        return GrpDict

    def GroupNames(self):
        # This function returns a list of the group names in the mesh
        if not hasattr(self,'__GroupInfo__'):
            self.__GroupSort__()

        return sum([list(item.keys()) for item in self.__GroupInfo__.values()], [])

    def GroupInfo(self, name, GroupType = None):
        if not hasattr(self,'__GroupInfo__'):
            self.__GroupSort__()

        grptype = [x for x in self.__GroupInfo__.keys() if name in self.__GroupInfo__[x].keys()]

        if len(grptype) == 0:
            err = 'Name not in mesh'
        elif len(grptype) == 1:
            grptype = grptype[0]
        elif len(grptype) > 1 and not GroupType:
            err = 'Multiple groups with same name and no type provided'
        elif len(grptype) > 1 and GroupType not in grptype:
            err = 'This is not one of the types of groups.'
        else :
            grptype = GroupType


        # Create empty class to assign information to
        class Groupdata:
            pass
        Groupdata.Type = grptype

        # Element group
        if grptype in ('NODE'):
            path = "{}/NOE".format(self.CnctPath)
            groupbool = np.in1d(self.g["{}/FAM".format(path)][:], self.__GroupInfo__[grptype][name])

            # Find indexes where groupbool is true then add 1 as nodes start from1 not 0 (only works if nodes are in order)
            Groupdata.Nodes = np.where(groupbool)[0] + 1
            Groupdata.NbNodes = len(Groupdata.Nodes)
            # not compatible with rmed files
#            Groupdata.Nodes = self.g["{}/NUM".format(path)][:][groupbool]

        # Element group
        else :
            path = "{}/MAI/{}".format(self.CnctPath,grptype)
            groupbool = np.in1d(self.g["{}/FAM".format(path)][:], self.__GroupInfo__[grptype][name])

            Groupdata.Elements = self.g["{}/NUM".format(path)][:][groupbool]
            Groupdata.NbElements = len(Groupdata.Elements)

            dset = self.g["{}/NOD".format(path)]
            Groupdata.Connect = np.reshape(dset[:], (dset.attrs['NBR'],int(dset.shape[0]/dset.attrs['NBR'])), order='F')[groupbool]
            Groupdata.Nodes = np.unique(Groupdata.Connect)
            Groupdata.NbNodes = len(Groupdata.Nodes)

        return Groupdata


    def GetNodeXYZ(self,nodes):
        nodelist = self.g["{}/NOE/COO".format(self.CnctPath)][:]
        # If individual node is supplied, i.e. of type int of np.int32/64
        if isinstance(nodes, (int, np.integer)):
            xyz = nodelist[np.array([nodes, nodes + self.NbNodes, nodes + 2*self.NbNodes]) -1]
        # If an array or list is passed
        else :
            NdCorrect = np.array(nodes) - 1
            xyz = nodelist[np.concatenate((NdCorrect, NdCorrect + self.NbNodes, NdCorrect + 2*self.NbNodes))]
            xyz = np.reshape(xyz, (len(nodes), 3), order = 'F')

        return xyz

    def Close(self):
        self.g.close()


def GroupData(ResFile,GroupName,ResName='Temperature'):
    g = h5py.File(ResFile, 'r')
    gRes = g['/CHA/{}'.format(ResName)]
    step = list(gRes.keys())[0]
    Result = gRes['{}/NOE/MED_NO_PROFILE_INTERNAL/CO'.format(step)][:]
    g.close()

    meshdata = MeshInfo(ResFile)
    GroupInfo = meshdata.GroupInfo(GroupName)
    NodeIDs = GroupInfo.Nodes
    GroupRes = Result[NodeIDs-1] # subtract 1 for 0 indexing
    return GroupRes




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
