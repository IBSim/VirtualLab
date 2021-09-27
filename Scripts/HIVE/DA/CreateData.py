import sys
import os
from natsort import natsorted
import h5py
from pathos.multiprocessing import ProcessPool
from importlib import import_module, reload
import numpy as np

from Functions import Uniformity2, Uniformity3
from Scripts.Common.VLFunctions import MeshInfo



def DataPool(VL, MLdict, ResDir):
    # function which is used by ProcessPool map
    print(ResDir)
    sys.path.insert(0,ResDir)
    Parameters = reload(import_module('Parameters'))
    sys.path.pop(0)

    ERMESres = h5py.File("{}/PreAster/ERMES.rmed".format(ResDir), 'r')
    attrs = ERMESres["EM_Load"].attrs
    Scale = (Parameters.Current/attrs['Current'])**2
    Watts = ERMESres["EM_Load/Watts"][:]*Scale
    JHNode =  ERMESres["EM_Load/JHNode"][:]*Scale
    ERMESres.close()
    # Calculate power
    CoilPower = np.sum(Watts)
    # Uniformity = Uniformity2(JHNode/Parameters.Current**2,"{}/PreAster/ERMES.rmed".format(ResDir) )
    Uniformity = Uniformity3(JHNode,"{}/PreAster/ERMES.rmed".format(ResDir) )

    Input = list(Parameters.CoilDisplacement) + [Parameters.Rotation]
    Output = [CoilPower,Uniformity]

    return Input+Output

def GetData(VL, MLdict, DataDirs, cores=9):
    NbDirs = len(DataDirs)
    Args = [[VL]*NbDirs,[MLdict]*NbDirs,DataDirs]
    Pool = ProcessPool(nodes=cores, workdir=VL.TEMP_DIR)
    Data = Pool.map(DataPool, *Args)
    return Data

def Single(VL, MLdict):
    ML = MLdict["Parameters"]

    # Master file where all data is stored
    DataFile_path = "{}/ML/Data.hdf5".format(VL.PROJECT_DIR)
    Database = h5py.File(DataFile_path,'r')
    DataPath = "{}/{}".format(ML.DataDir,ML.DataName)
    if DataPath in Database:
        CurrentData = Database[DataPath][:]
        dsetExist = True
    else : dsetExist = False
    Database.close()

    # Check what we'll be doing



    if dsetExist and ML.AddData.lower() == 'overwrite':
        print("Data in {} will be overwritten with new data".format(DataPath))
    elif dsetExist and  ML.AddData.lower() in ('append','appendall'):
        print("Data in {} will be appended with new data".format(DataPath))
    elif dsetExist:
        print('No new data to include')
        return
    else: #dsetExist is False
        print("New dataset {} will be created".format(DataPath))
        Append=AppendAll=False
        Overwrite=True

    # Get data
    if hasattr(ML,'Data'):
        DataArr = ML.Data # This needs to be an array of data to add to file
    else :
        DataDir = "{}/{}".format(VL.PROJECT_DIR,ML.DataDir)
        DataDir_Sub = []
        for _dir in os.listdir(DataDir):
            _path = "{}/{}".format(DataDir,_dir)
            if os.path.isdir(_path): DataDir_Sub.append(_path)
        DataDir_Sub = natsorted(DataDir_Sub)

        if ML.AddData.lower() == 'append': #only append new
            if len(DataDir_Sub) <= CurrentData.shape[0]:
                print("No new data to append - consider AppendAll")
                return
            else :
                DataDir_Sub = DataDir_Sub[CurrentData.shape[0]:]

        DataArr = GetData(VL,MLdict,DataDir_Sub)

    if ML.AddData.lower() in ('append','appendall'):
        DataArr = np.vstack((CurrentData,DataArr))

    Database = h5py.File(DataFile_path,'a')
    if dsetExist: del Database[DataPath]
    Database.create_dataset(DataPath, data=DataArr, maxshape=(None,None))
    Database.close()
