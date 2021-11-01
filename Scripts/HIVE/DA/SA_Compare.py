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

def implotter(data,label,title,loc,fname):

    fnt=12
    for i,tp in enumerate(['Power','MaxTemp']):

        fig, axes = plt.subplots(nrows=2,ncols=1, figsize=(10,10))
        ln = []
        for dat,lab in zip(data,label):
            axes[0].plot(dat[:,i+2], label=lab)
            if dat.shape[0] > len(ln):
                ln = dat[:,0].tolist()

        axes[0].legend(fontsize=fnt)
        axes[0].set_xlabel('Mesh Fineness',fontsize=fnt)
        axes[0].set_ylabel(tp,fontsize=fnt)

        axes[0].set_xticks(range(len(ln)))
        axes[0].set_xticklabels(ln,fontsize=fnt)

        for dat,lab in zip(data,label):
            axes[1].plot(dat[:,1],dat[:,i+2], label=lab)
        axes[1].legend(fontsize=fnt)
        axes[1].set_xlabel('NbNodes',fontsize=fnt)
        axes[1].set_ylabel(tp,fontsize=fnt)

        fig.suptitle(title,fontsize=fnt)

        outdir = "{}/{}".format(loc,tp)
        os.makedirs(outdir,exist_ok=True)
        plt.savefig("{}/{}.png".format(outdir,fname),dpi=200)
        plt.close()

def Single(VL, DADict):
    Parameters = DADict["Parameters"]

    ResDir = "{}/{}".format(VL.PROJECT_DIR, DADict["Name"])

    ResDict = {}
    for dir in os.listdir(VL.PROJECT_DIR):
        if dir in ('Meshes','.log',Parameters.Name): continue
        if dir.startswith('_'):continue
        _resdir = "{}/{}".format(VL.PROJECT_DIR,dir)
        ResDict[dir] = []
        for resname in natsorted(os.listdir(_resdir)):
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

            Watts = JH_Vol*Volumes
            Power = Watts.sum()

            resfile = "{}/Aster/Thermal.rmed".format(resdir)

            CAres = h5py.File(resfile, 'r')
            pth = 'CHA/Temperature/0000000000000000000000000000000000000000/NOE/MED_NO_PROFILE_INTERNAL/CO'
            temps = CAres[pth][:]
            maxtemp = temps.max()
            print(resname,maxtemp)

            ResDict[dir].append([MeshParam.SubTile,meshdata.NbNodes,Power,maxtemp])

    for key, val in ResDict.items():
        npval = np.array(val)
        print(key)
        print(npval)
        ResDict[key]=npval

    implotter([ResDict['NoFillet']],
              ['NoFillet'],
              'No Fillet',
              DADict['CALC_DIR'],
              'NoFillet')

    implotter([ResDict['Fillet_0.0001']],
              ['Fillet'],
              'Fillet',
              DADict['CALC_DIR'],
              'Fillet')

    implotter([ResDict['Fillet_0.0005'],ResDict['Fillet_0.0001']],
              ['Fillet_0.0005','Fillet_0.0001'],
              'Fillet size',
              DADict['CALC_DIR'],
              'FilletSize')

    implotter([ResDict['Fillet_0.0001'],ResDict['Fillet_0.0001_Fine']],
              ['Fillet_2','Fillet_5'],
             'Fillet Fineness',
              DADict['CALC_DIR'],
              'FilletFineness')

    implotter([ResDict['Fillet_0.0001'],ResDict['TopFace'],ResDict['TopFace_Fine']],
              ['Fillet','TopFace','TopFace_Fine'],
             'TopFace Refinement',
              DADict['CALC_DIR'],
              'TopFaceRefinement')



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
