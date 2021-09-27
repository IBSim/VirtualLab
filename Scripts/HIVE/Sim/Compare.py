import h5py
import os
import numpy as np
import matplotlib.pyplot as plt

def Combined(Info):
    Compare_Dir = "{}/Comparison".format(Info.STUDY_DIR)
    if not os.path.isdir(Compare_Dir): os.makedirs(Compare_Dir)

    SimMaster = Info.Parameters_Master.Sim
    Benchmark = "{}/{}/Aster/ResTher.rmed".format(Info.STUDY_DIR,SimMaster.Benchmark)
    Full = h5py.File(Benchmark, 'r')
    FullRes = Full["CHA/resther_TEMP"]
    ResDict = {}
    MaxDiff = {}
    Mean = {SimMaster.Benchmark:[]}
    for Name, StudyDict in Info.Studies.items():
        ResDict[Name] = h5py.File("{}/ResTher.rmed".format(StudyDict["ASTER"]), 'r')
        MaxDiff[Name] = []
        Mean[Name] = []

    Time = []
    for step in FullRes.keys():
        Time.append(FullRes[step].attrs["PDT"])
        dsetFull = FullRes["{}/NOE/MED_NO_PROFILE_INTERNAL/CO".format(step)][:]
        Mean[SimMaster.Benchmark].append(np.mean(dsetFull))
        for Name in Info.Studies.keys():
            StudyRes = ResDict[Name]["CHA/resther_TEMP"]
            if step not in StudyRes.keys(): continue

            dsetStudy = ResDict[Name]["CHA/resther_TEMP/{}/NOE/MED_NO_PROFILE_INTERNAL/CO".format(step)][:]
            Diff = dsetStudy - dsetFull
            # if Name == 'TestCoil09':
                # print(np.mean(dsetStudy))
            MaxDiff[Name].append(np.max(np.abs(Diff)))
            Mean[Name].append(np.mean(dsetStudy))

    for Name, val in Mean.items():
        print(Name, val[-1])

    fig = plt.figure(figsize = (14,5))
    plt.xlabel('Time',fontsize = 20)
    plt.ylabel('Average temperature',fontsize = 20)
    # plt.plot(Time,Mean[SimMaster.Benchmark])
    for Name, Res in Mean.items():
        plt.plot(Time[:len(Res)], Res, label = Name)
    plt.legend(loc='upper left')
    plt.show()


    fig = plt.figure(figsize = (14,5))
    plt.xlabel('Time',fontsize = 20)
    plt.ylabel('Max Temp Change',fontsize = 20)
    for Name, Res in MaxDiff.items():
        plt.plot(Time[:len(Res)], Res, label = Name)
    plt.legend(loc='upper left')
    plt.show()



    # for Name, StudyDict in Info.Studies.items():
