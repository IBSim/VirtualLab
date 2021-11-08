import numpy as np
import os
import matplotlib.pyplot as plt
from importlib import import_module, reload
import sys
import h5py
import scipy
import pickle

from natsort import natsorted
from VLFunctions import MeshInfo

def implotter(data,label,title,ylabel,loc):

    fnt=12

    fig, axes = plt.subplots(nrows=2,ncols=1, figsize=(10,10))
    ln = []
    for dat,lab in zip(data,label):
        axes[0].plot(dat[:,-1], label=lab)
        if dat.shape[0] > len(ln):
            ln = dat[:,0].tolist()

    axes[0].legend(fontsize=fnt)
    axes[0].set_xlabel('Mesh Fineness',fontsize=fnt)
    axes[0].set_ylabel(ylabel,fontsize=fnt)

    axes[0].set_xticks(range(len(ln)))
    axes[0].set_xticklabels(ln,fontsize=fnt)

    for dat,lab in zip(data,label):
        axes[1].plot(dat[:,1],dat[:,-1], marker='x',label=lab)
    axes[1].legend(fontsize=fnt)
    axes[1].set_xlabel('NbNodes',fontsize=fnt)
    axes[1].set_ylabel(ylabel,fontsize=fnt)

    fig.suptitle(title,fontsize=fnt)

    os.makedirs(os.path.dirname(loc),exist_ok=True)
    plt.savefig("{}.png".format(loc),dpi=200)
    plt.close()

def implotter2(data,label,title,ylabel,loc):
    fnt=12

    fig, axes = plt.subplots(nrows=1,ncols=1, figsize=(10,10))
    ln = []
    axes.plot(data[0][:,1],data[0][:,-1], label=label[0])
    for dat,lab in zip(data[1:],label[1:]):
        axes.scatter(dat[:,1],dat[:,-1], label=lab)
        if dat.shape[0] > len(ln):
            ln = dat[:,0].tolist()

    axes.legend(fontsize=fnt)
    axes.set_xlabel('NbNodes',fontsize=fnt)
    axes.set_ylabel(ylabel,fontsize=fnt)

    fig.suptitle(title,fontsize=fnt)

    os.makedirs(os.path.dirname(loc),exist_ok=True)
    plt.savefig("{}.png".format(loc),dpi=200)
    plt.close()

def Single(VL, DADict):
    Parameters = DADict["Parameters"]

    ResDir = "{}/{}".format(VL.PROJECT_DIR, DADict["Name"])

    ResERMES,ResCA = {},{}
    for dir in os.listdir(VL.PROJECT_DIR):
        if dir in ('Meshes','.log',Parameters.Name): continue
        if dir.startswith('_'):continue
        _resdir = "{}/{}".format(VL.PROJECT_DIR,dir)
        ResERMES[dir], ResCA[dir] = [],[]
        for resname in natsorted(os.listdir(_resdir)):
            if resname.startswith("_"): continue
            # Get parameters
            resdir = "{}/{}".format(_resdir,resname)
            with open("{}/.Parameters.pkl".format(resdir),'rb') as f:
                ResParam = pickle.load(f)

            Meshfname = "{}/{}".format(VL.MESH_DIR,ResParam.Mesh)
            Meshfile = "{}.med".format(Meshfname)
            meshdata = MeshInfo(Meshfile)
            meshdata.Close()

            pklfile = "{}/.{}.pkl".format(os.path.dirname(Meshfname),
                                    os.path.basename(Meshfname))
            with open(pklfile,'rb') as f:
                MeshParam = pickle.load(f)

            ERMESresfile = "{}/PreAster/ERMES.rmed".format(resdir)
            ERMESres = h5py.File(ERMESresfile, 'r')
            Volumes = ERMESres["EM_Load/Volumes"][:]
            JH_Vol = ERMESres["EM_Load/JH_Vol"][:]
            ERMESres.close()

            EMmeshdata = MeshInfo(ERMESresfile)
            EMmeshdata.Close()

            Watts = JH_Vol*Volumes
            Power = Watts.sum()

            ResERMES[dir].append([MeshParam.SubTile,meshdata.NbNodes,Power])

            resfile = "{}/Aster/Thermal.rmed".format(resdir)
            CAres = h5py.File(resfile, 'r')
            pth = 'CHA/Temperature/0000000000000000000000000000000000000000/NOE/MED_NO_PROFILE_INTERNAL/CO'
            temps = CAres[pth][:]
            maxtemp = temps.max()
            print(resname,maxtemp)

            ResCA[dir].append([MeshParam.SubTile,meshdata.NbNodes,maxtemp])

    for key in ResCA:
        ResCA[key]=np.array(ResCA[key])
        ResERMES[key]=np.array(ResERMES[key])


    for dc,path in [[ResERMES,'Power'],[ResCA,'MaxTemp']]:
        implotter([dc['NoFillet']],
                  ['NoFillet'],
                  'No Fillet',
                  path,
                  "{}/{}/NoFillet".format(DADict['CALC_DIR'],path))

        implotter([dc['Fillet_0.0001']],
                  ['Fillet'],
                  'Fillet',
                  path,
                  "{}/{}/Fillet".format(DADict['CALC_DIR'],path))

        implotter([dc['Fillet_0.0005'],dc['Fillet_0.0001']],
                  ['Fillet_0.0005','Fillet_0.0001'],
                  'Fillet size',
                  path,
                  "{}/{}/FilletSize".format(DADict['CALC_DIR'],path))

        implotter([dc['Fillet_0.0001_Coarse'],dc['Fillet_0.0001'][:3,:],dc['Fillet_0.0001_Fine']],
                  ['Fillet_1','Fillet_2','Fillet_5'],
                 'Fillet Fineness',
                 path,
                 "{}/{}/FilletFineness".format(DADict['CALC_DIR'],path))

        implotter([dc['Fillet_0.0001'],dc['TopFace'],dc['TopFace_Fine']],
                  ['Fillet','TopFace','TopFace_Fine'],
                 'TopFace Refinement',
                 path,
                 "{}/{}/TopFaceRefinement".format(DADict['CALC_DIR'],path))

        implotter2([dc['Fillet_0.0001'][:5,:],dc['MR_NFace_1'],dc['MR_NFace_3'],dc['MR_MFace_1']],
                  ['Fillet','MR_0.001','MR_0.0006','MR2_0.001'],
                 'Mesh Refinement',
                 path,
                 "{}/{}/MeshRefinement".format(DADict['CALC_DIR'],path))


    #
    # implotter([ResDict['Fillet_C_0.0001'],ResDict['Fillet_C_0.0001_CF_0.0005'],ResDict['TopSide']],
    #           ['NoFaceRefine','Top','TopAndSide'],
    #          'Face Refinements',
    #           DADict['CALC_DIR'],
    #           'FaceRefinement')

    # implotter([ResDict['Fillet_C_0.0001'],ResDict['Master']],
    #           ['Fillet','Refined'],
    #          'Master',
    #           DADict['CALC_DIR'],
    #           'Master')
