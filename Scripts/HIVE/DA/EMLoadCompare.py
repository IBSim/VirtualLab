import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from importlib import import_module, reload
import sys
import re
import h5py

def Single(VL, MLdict):
    Parameters = MLdict["Parameters"]

    ResDir = "{}/{}".format(VL.PROJECT_DIR, MLdict["Name"])

    Benchmark = h5py.File("{}/Sim_1_0/Aster/ResTher.rmed".format(ResDir),'r')
    BRes = Benchmark["CHA/resther_TEMP"]
    Btime,Bkeys = [],[]
    for key,val in BRes.items():
        Btime.append(val.attrs['PDT'])
        Bkeys.append(key)

    Convert = {0.9:2039,0.99:3644,0.999:5593,1:21323}

    Threshold,NbClusters = [],[]
    TimeData,ResDiff = [],[]
    for ResName in os.listdir(ResDir):
        ResSubDir = "{}/{}".format(ResDir,ResName)

        if ResName.startswith('_'): continue
        if not os.path.isdir(ResSubDir): continue

        # import parameters used for Simulation
        sys.path.insert(0,ResSubDir)
        SimParameters = reload(import_module('Parameters'))
        sys.path.pop(0)

        _NbClusters,_Threshold = SimParameters.NbClusters,SimParameters.Threshold

        # Currently only plot for full threshold or no clusters
        if not (_NbClusters==0 or _Threshold==1): continue
        # if _NbClusters !=0: continue
        # print(_NbClusters)

        Threshold.append(_Threshold)
        NbClusters.append(_NbClusters)

        AsterLog = "{}/Aster/AsterLog".format(ResSubDir)
        with open(AsterLog,'r') as f:
            app=False
            for line in f:
                if line.startswith(' * COMMAND'): app=True
                if not app:continue
                split = line.split()
                if len(split)<=1: continue
                timeax = -4
                if split[1]=='LIRE_MAILLAGE': mesh = float(split[timeax])
                if split[1]=='AFFE_CHAR_THER': setup = float(split[timeax])
                if split[1]=='TOTAL_JOB':
                    total = float(split[timeax])
                    break
        TimeData.append([mesh,setup,total])

        Simres = h5py.File("{}/Aster/ResTher.rmed".format(ResSubDir),'r')
        Difflst = []
        for val in Simres["CHA/resther_TEMP"].values():
            t = val.attrs['PDT']
            if t in Btime:
                Bkey = Bkeys[Btime.index(t)]
                Barr = BRes["{}/NOE/MED_NO_PROFILE_INTERNAL/CO".format(Bkey)][:]
                arr = val["NOE/MED_NO_PROFILE_INTERNAL/CO"][:]
                maxdiff = np.abs(Barr - arr).max()
                Difflst.append(maxdiff)

        ResDiff.append(Difflst)
        Simres.close()


    Benchmark.close()

    _Threshold,NbClusters = np.array(Threshold),np.array(NbClusters)
    TimeData,ResDiff = np.array(TimeData),np.array(ResDiff)
    Threshold = np.array([Convert[a] for a in _Threshold])

    # fig, (ax1,ax2,ax3) = plt.subplots(nrows=3,ncols=1,figsize=(15,10))
    fig = plt.figure(constrained_layout=False,figsize=(15,10))
    fig.suptitle('Threshold effect')

    gs = GridSpec(2, 2, figure=fig)
    ax1=fig.add_subplot(gs[0,0])
    ax2=fig.add_subplot(gs[0,1])
    ax3=fig.add_subplot(gs[1,:])

    Ixs = (NbClusters==0).nonzero()[0]
    Ixs = Ixs[np.argsort(_Threshold[Ixs])]

    arr = TimeData[Ixs]
    x_pos = np.arange(arr.shape[0])

    totalsetup = arr[:,0]+arr[:,1]
    ax1.plot(x_pos,arr[:,0],label='Read Mesh')
    ax1.plot(x_pos,arr[:,1],label='ApplyBB')
    ax1.plot(x_pos,totalsetup,label='TotalSetup')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(Threshold[Ixs])
    ax1.set_ylabel('Time (s)')
    ax1.set_xlabel('Threshold')
    ax1.legend()

    solving = arr[:,2] - (totalsetup)
    ax2.plot(x_pos,solving,label='Solver')
    ax2.plot(x_pos,arr[:,2],label='Total')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(Threshold[Ixs])
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Time (s)')
    ax2.legend()

    for _thold, thold, res in zip(_Threshold[Ixs],Threshold[Ixs],ResDiff[Ixs]):
        ax3.plot(Btime,res,label="{} ({})".format(thold,_thold))
    ax3.set_xlabel('Simulation Time')
    ax3.set_ylabel('Max Difference')
    ax3.legend()

    plt.savefig("{}/Thresholding.png".format(ResDir))
    plt.close()

    fig = plt.figure(constrained_layout=False,figsize=(15,10))
    fig.suptitle('Clustering effect')

    gs = GridSpec(2, 2, figure=fig)
    ax1=fig.add_subplot(gs[0,0])
    ax2=fig.add_subplot(gs[0,1])
    ax3=fig.add_subplot(gs[1,:])

    Ixs = (_Threshold==1).nonzero()[0]
    Ixs = Ixs[np.argsort(NbClusters[Ixs])]
    Ixs = np.roll(Ixs,-1)

    arr = TimeData[Ixs]
    x_pos = np.arange(arr.shape[0])

    totalsetup = arr[:,0]+arr[:,1]
    ax1.plot(x_pos,arr[:,0],label='Read Mesh')
    ax1.plot(x_pos,arr[:,1],label='ApplyBB')
    ax1.plot(x_pos,totalsetup,label='TotalSetup')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(NbClusters[Ixs])
    ax1.set_ylabel('Time (s)')
    ax1.set_xlabel('Clusters')
    ax1.legend()

    solving = arr[:,2] - (totalsetup)
    ax2.plot(x_pos,solving,label='Solver')
    ax2.plot(x_pos,arr[:,2],label='Total')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(NbClusters[Ixs])
    ax2.set_xlabel('Clusters')
    ax2.set_ylabel('Time (s)')
    ax2.legend()

    for cls,res in zip(NbClusters[Ixs],ResDiff[Ixs]):
        ax3.plot(Btime,res,label=cls)
    ax3.legend()

    plt.savefig("{}/Clustering.png".format(ResDir))
    plt.close()






    #
