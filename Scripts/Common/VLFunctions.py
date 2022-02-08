import h5py
import sys
import numpy as np
import os
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace as Namespace
from itertools import product
from scipy import special
import pickle
from importlib import import_module, reload

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



class Sampling():
    def __init__(self, method, dim=0, range=[], bounds=True, seed=None,options={}):
        # Must have either a range or dimension
        if range:
            self.range = range
            self.dim = len(range)
        elif dim:
            self.range = [(0,1)]*dim
            self.dim = dim
        else:
            print('Error: Must provide either dimension or range')

        self.options = options
        self.bounds = bounds
        self._boundscomplete = False
        self._Nb = 0

        if method.lower() == 'halton': self.sampler = self.Halton
        elif method.lower() == 'random':
            if seed: np.random.seed(seed)
            self.sampler = self.Random
        elif method.lower() == 'sobol': self.sampler = self.Sobol
        elif method.lower() == 'grid': self.sampler = self.Grid
        elif method.lower() == 'subspace': self.sampler = self.SubSpace


    def get(self,N):
        self.N=N
        norm = self.sampler()
        range = np.array(self.range)
        scale = norm*(range[:,1] - range[:,0]) + range[:,0]
        return scale.T.tolist()

    def getbounds(self):
        if self.bounds and not self._boundscomplete:
            Bnds = np.array(list(zip(*product(*[[0,1]]*self.dim)))).T
            Bnds = Bnds[self._Nb:self._Nb + self.N]
            self._Nb += Bnds.shape[0]
            if self._Nb < 2**self.dim:
                self.N=0
            else:
                self._boundscomplete=True
                self.N -= Bnds.shape[0]
            return Bnds

    def Halton(self):
        import ghalton
        if not hasattr(self,'generator'):
            self.generator = ghalton.Halton(self.dim)

        Bnds = self.getbounds()
        if self.N == 0: return Bnds

        Points = np.array(self.generator.get(self.N))

        if isinstance(Bnds,np.ndarray):
            return np.vstack((Bnds,Points))
        else :
            return Points

    def Random(self):
        Bnds = self.getbounds()
        if self.N == 0: return Bnds

        Points = np.random.uniform(size=(self.N,self.dim))
        if isinstance(Bnds,np.ndarray):
            return np.vstack((Bnds,Points))
        else :
            return Points

    def Sobol(self):
        import torch
        if not hasattr(self,'generator'):
            self.generator = torch.quasirandom.SobolEngine(dimension=self.dim)

        Bnds = self.getbounds()
        if self.N == 0: return Bnds

        Points = self.generator.draw(self.N).detach().numpy().tolist()
        if isinstance(Bnds,np.ndarray):
            return np.vstack((Bnds,Points))
        else :
            return Points

    def Grid(self):
        disc = np.linspace(0,1,self.N)
        return np.array(list(zip(*product(*[disc]*self.dim)))).T

    def _SubSpace_dist(self, N, f, d):
        Dist = []
        for i in range(d):
            SS_N = int(np.ceil(N*f))
            Dist.append(SS_N)
            N -= SS_N
        Dist.append(N)
        return Dist

    def _SubSpace_fn(self,N):
        # This function has looked at empricial results and seems to match well
        # with curves for 2,3 and 4 D
        n = N**(1/self.dim) # self.dim th root of N
        return (1 - 1/n)**2

    def _SubSpace(self):
        # defaults which can be updated using options
        options = {'dim': self.dim - 1, # number of dimensions down to survey
                   'method': 'halton',
                   'roll' : True,
                   'SS_Func' : self._SubSpace_fn}
        options.update(self.options)

        # Setup
        d = options['dim']
        if not hasattr(self,'generator'):
            self.Add = 0
            self.SumNb = np.array([0]*(d+1))
            self.IndNb = np.array([0]*(d+1))
            self.generator = [[] for _ in range(self.dim)]

            self._generator,self.SS = [],[]
            if options['method'].lower() == 'halton':
                import ghalton
                for i in range(d+1):
                    self._generator.append(ghalton.GeneralizedHalton(self.dim,0))
                    self.SS.append(special.binom(self.dim,self.dim-i)*2**i)

            elif options['method'].lower() == 'sobol':
                import torch
                for i in range(d+1):
                    self._generator.append(torch.quasirandom.SobolEngine(dimension=self.dim))
                    self.SS.append(special.binom(self.dim,self.dim-i)*2**i)

        # work out how many points from each subspace we will need
        fact = options['SS_Func'](self.Add+self.N)
        fact = max(fact,0.1) # ensures fact is never zero
        Dist = self._SubSpace_dist(self.Add+self.N,fact,d)
        # Divide by number of subspaces for each dimension & work out difference
        Ind = np.ceil(np.array(Dist)/self.SS) - self.IndNb
        # Get the necessary points needed and add to 'generator'
        for i, (num, gen) in enumerate(zip(Ind,self._generator)):
            if num == 0: continue
            # get num points from generator
            if options['method'].lower() == 'halton':
                genvals = gen.get(int(num))
            elif options['method'].lower() == 'sobol':
                genvals = gen.draw(int(num)).detach().numpy().tolist()
            self.IndNb[i]+=num # keep track of how many of each dim we've asked for

            if i==0:
                points = genvals
            else:
                points = []
                BndIxs = list(combinations(list(range(self.dim)),i)) # n chose i
                BndPerms = list(product(*[[0,1]]*i))
                for genval,BndIx,BndPerm in product(genvals,BndIxs,BndPerms):
                    genval = np.array(genval)
                    # roll generator to put bases out of sync (optional)
                    if options['roll']:
                        genval = np.roll(genval,i)

                    genval[list(BndIx)] = list(BndPerm)
                    points.append(genval.tolist())

            self.generator[i].extend(points)

        # Get N points one at a time from self.generator
        # to ensures ordering is maintained
        points = []
        for i in range(1,self.N+1):
            _N = self.Add+i
            fact = options['SS_Func'](_N)
            fact = max(fact,0.1) # ensures fact is never zero

            Dist = self._SubSpace_dist(_N,fact,d)
            # Get index of next point
            ix = (np.array(Dist) - self.SumNb).argmax()
            point = self.generator[ix].pop(0)
            points.append(point)
            self.SumNb[ix]+=1

        self.Add+=self.N

        return points

    def SubSpace(self):
        Bnds = self.getbounds()
        if self.N == 0: return Bnds

        Points = self._SubSpace()

        if isinstance(Bnds,np.ndarray):
            return np.vstack((Bnds,Points))
        else :
            return Points
