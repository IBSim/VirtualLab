import sys
import os
from natsort import natsorted
import h5py
from pathos.multiprocessing import ProcessPool
from importlib import import_module, reload
import numpy as np
from Scripts.Common.VLFunctions import MeshInfo
from scipy import spatial

def DataPool(VL, MLdict, ResDir):
    # function which is used by ProcessPool map
    sys.path.insert(0,ResDir)
    Parameters = reload(import_module('Parameters'))
    sys.path.pop(0)

    ERMESres = h5py.File("{}/PreAster/ERMES.rmed".format(ResDir), 'r')
    Watts = ERMESres["EM_Load/Watts"][:]
    JHNode =  ERMESres["EM_Load/JHNode"][:]
    ERMESres.close()
    # Calculate power
    CoilPower = np.sum(Watts)

    tst = np.inf
    JHNode /= Parameters.Current **2
    # Meshcls = MeshInfo("{}/{}.med".format(MeshDir,Parameters.Mesh))
    Meshcls = MeshInfo("{}/PreAster/ERMES.rmed".format(ResDir))
    CoilFace = Meshcls.GroupInfo('CoilFace')
    Area, JHArea = 0, 0 # Actual area and area of triangles with JH
    for nodes in CoilFace.Connect:
        vertices = Meshcls.GetNodeXYZ(nodes)
        # Heron's formula
        a, b, c = spatial.distance.pdist(vertices, metric="euclidean")
        s = 0.5 * (a + b + c)
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        Area += area

        vertices[:,2] += JHNode[nodes - 1].flatten()

        a, b, c = spatial.distance.pdist(vertices, metric="euclidean")
        s = 0.5 * (a + b + c)
        area1 = np.sqrt(s * (s - a) * (s - b) * (s - c))
        JHArea += area1

    Meshcls.Close()
    Uniformity = JHArea/Area

    Input = list(Parameters.CoilDisplacement) + [Parameters.Rotation]
    Output = [CoilPower,Uniformity]

    return Input+Output

def GetData(VL, MLdict, DataDirs, cores=5):
    NbDirs = len(DataDirs)
    Args = [[VL]*NbDirs,[MLdict]*NbDirs,DataDirs]
    Pool = ProcessPool(nodes=cores, workdir=VL.TEMP_DIR)
    Data = Pool.map(DataPool, *Args)
    return Data

def Single(VL, MLdict):
    ML = MLdict["Parameters"]
    DataDir = "{}/{}".format(VL.PROJECT_DIR,ML.DataDir)
    # Get only sub directories in DataDir & sort them using natsort
    DataDir_Sub = []
    for _dir in os.listdir(DataDir):
        _path = "{}/{}".format(DataDir,_dir)
        if os.path.isdir(_path): DataDir_Sub.append(_path)
    DataDir_Sub = natsorted(DataDir_Sub)

    # File where all data is stored
    DataFile_path = "{}/Data.hdf5".format(VL.ML_DIR)
    Database = h5py.File(DataFile_path,'r')
    DataPath = "{}/{}".format(ML.DataDir,ML.DataName)

    if DataPath in Database:
        CurrentData = Database[DataPath][:]
        dsetExist = True
    else : dsetExist = False
    Database.close()

    Overwrite = getattr(ML,'Overwrite', False)
    AppendAll = getattr(ML,'AppendAll', False)

    Write = True
    if dsetExist and Overwrite:
        print("Data will be overwritten")
        DataArr = GetData(VL,MLdict,DataDir_Sub)
    elif dsetExist and AppendAll:
        print("All data will be appended")
        DataArr = GetData(VL,MLdict,DataDir_Sub)
        DataArr = np.vstack((CurrentData,DataArr))
    elif dsetExist and len(DataDir_Sub) > CurrentData.shape[0]:
        print("New data will be appended")
        NbPoints = CurrentData.shape[0]
        print(len(DataDir_Sub) - NbPoints)
        DataArr = GetData(VL,MLdict,DataDir_Sub[NbPoints:])
        DataArr = np.vstack((CurrentData,DataArr))
    elif dsetExist:
        print('No new data to include')
        Write = False
    else: #dsetExist is False
        print("New dataset {} will be added".format(DataPath))
        DataArr = GetData(VL,MLdict,DataDir_Sub)

    if Write:
        Database = h5py.File(DataFile_path,'a')
        if dsetExist: del Database[DataPath]
        Database.create_dataset(DataPath, data=DataArr, maxshape=(None,None))
        Database.close()
