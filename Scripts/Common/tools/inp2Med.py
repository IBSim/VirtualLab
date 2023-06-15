import os
import sys
import h5py
import numpy as np
current_dir = os.path.dirname(os.path.abspath(__file__))

def main(mesh_in):

    if mesh_in.startswith('/'):
        mesh_path = mesh_in # mesh is full path
    else:
        mesh_path = "{}/{}".format(os.getcwd(),mesh_in)

    if not os.path.isfile(mesh_path):
        sys.exit('Mesh {} does not exist'.format(mesh_path))

    fname, ext = os.path.splitext(mesh_path)
    MeshName = os.path.basename(fname)
    if ext != '.inp':
        sys.exit('Mesh does not have inp extension')

    print("Converting mesh {} from inp to med\n".format(MeshName))

    fileout = '{}.med'.format(fname)

    f = open(mesh_path,'r')
    flines = f.readlines()
    f.close()
    NumLin = len(flines)
    sum1 = 0
    Section, Nvols = '', 0
    NdNum, Coords = [], []
    VGroupInfo = []
    EGrupInfo, SGroupInfo, NGroupInfo = {}, {}, {}
    for i,line in enumerate(flines):
        if line[0:2] == '**':
            if line.startswith('**NODE DATA BEGIN'):
                Section = 'Node'
            elif line.startswith('**NODE DATA END'):
                Section = 'NodeFin'

            elif line.startswith('**SOLID ELEMENT DATA BEGIN'):
                Section = 'Elem'
            elif line.startswith('**SOLID ELEMENT DATA END'):
                Section = 'ElemFin'

            elif line.startswith('**CONTACT SURFACE NODE SET BEGIN'):
                Section = 'NGroup'
                NGrouptmp = []
            elif line.startswith('**CONTACT SURFACE NODE SET END'):
                Section = 'NGroupFin'
                NGroupInfo[(Ngrpname,)] = NGrouptmp

            elif line.startswith('**CONTACT SURFACE ELEMENTS BEGIN'):
                Section = 'SGroup'
                SGrouptmp = []
            elif line.startswith('**CONTACT SURFACE ELEMENTS END'):
                Section = 'SGroupFin'
                SGroupInfo[(Sgrpname,)] = SGrouptmp

        if Section == 'Node':
            data = line.replace(',',' ').split()
            if data[0].isnumeric():
                NdNum.append(int(data[0]))
                Coords.append([float(data[1]),float(data[2]),float(data[3])])

        if Section == 'Elem':
            data = line.replace(',',' ').split()

            if data[0] == '**ELEMENTS':
                if data[2] == 'Part:' and data[-1] == 'BEGIN':
                    elgrpname = data[3]
                    Ellist, Conlist = [],[]
                elif data[2] == 'Part:' and data[-1] == 'END':
                    gpels = len(Ellist)
#                    print('group {} has {} volumes'.format(elgrpname,gpels))
                    Nvols += gpels
                    VGroupInfo.append(([elgrpname],Ellist,Conlist))

            elif data[0].isnumeric():
                Ellist.append(int(data[0]))
                Conlist.append([int(data[1]),int(data[2]),int(data[3]),int(data[4])])

        if Section == 'NGroup':
            if line[0:2] == '**':
                if line.startswith('**Contact node set'):
                    Ngrpname = line[20:].strip()

            data = line.replace(',',' ').split()
            if data[0].isnumeric():
                NGrouptmp += list(map(int,data))

        if Section == 'SGroup':
            if line[0:2] == '**':
                if line.startswith('**Contact'):
                    Sgrpname = line[11:].strip()

            data = line.replace(',',' ').split()
            if data[0].isnumeric():
                SGrouptmp.append((int(data[0]),int(data[1][1:])))



    # Write mesh information in to .med format
    f = h5py.File(fileout, 'w')
    Ver = '4.0.0'

    IG = f.create_group('INFOS_GENERALES')
    IG.attrs.create('MAJ', Ver[0],dtype = 'i4')
    IG.attrs.create('MIN', Ver[2],dtype = 'i4')
    IG.attrs.create('REL', Ver[4],dtype = 'i4')

    grp = f.create_group('/ENS_MAA/{}'.format(MeshName))
    grp.attrs.create('DES','',dtype = 'S1')
    grp.attrs.create('DIM',3,dtype = 'i4')
    grp.attrs.create('ESP',3,dtype = 'i4')
    grp.attrs.create('NOM','',dtype = 'S1')
    grp.attrs.create('NXI',-1,dtype = 'i4')
    grp.attrs.create('NXT',-1,dtype = 'i4')
    grp.attrs.create('REP',0,dtype = 'i4')
    grp.attrs.create('SRT',0,dtype = 'i4')
    grp.attrs.create('TYP',0,dtype = 'i4')
    grp.attrs.create('UNI','',dtype = 'S1')
    grp.attrs.create('UNT','',dtype = 'S1')

    grp = grp.create_group('-0000000000000000001-0000000000000000001')
    grp.attrs.create('CGT',1,dtype='i4')
    grp.attrs.create('NDT',-1,dtype='i4')
    grp.attrs.create('NOR',-1,dtype='i4')
    grp.attrs.create('NXI',-1,dtype = 'i4')
    grp.attrs.create('NXT',-1,dtype = 'i4')
    grp.attrs.create('PDT',-1.0,dtype='f8')
    grp.attrs.create('PVI',-1,dtype='i4')
    grp.attrs.create('PVT',-1,dtype='i4')

    Elems = grp.create_group('MAI')
    Elems.attrs.create('CGT',1,dtype='i4')

    Nodes = grp.create_group('NOE')
    Nodes.attrs.create('CGS',1,dtype='i4')
    Nodes.attrs.create('CGT',1,dtype='i4')
    Nodes.attrs.create('PFL','MED_NO_PROFILE_INTERNAL',dtype='S100')


    FAS = f.create_group('/FAS/{}'.format(MeshName))
    grp = FAS.create_group('FAMILLE_ZERO')
    grp.attrs.create('NUM',0,dtype='i4')


    g = h5py.File('{}/MED_Format.med'.format(current_dir), 'r')


    ##### Elements
    f.copy(g['ELEME'],'FAS/{}/ELEME/'.format(MeshName))
    ELEME = f['FAS/{}/ELEME/'.format(MeshName)]

    ### Look at updating the volume part
    # Volume groups
    VArr = np.zeros(Nvols,dtype='i4')
    NumSt, Elgrp = 0, -1
    ElNum, Cnct = [], []
    for name, num, connec in VGroupInfo:
        # Create the group information
        grp = ELEME.create_group('grp{}'.format(Elgrp))
        grp.attrs.create('NUM',Elgrp,dtype='i4')
        grp = grp.create_group('GRO')
        grp.attrs.create('NBR',1,dtype='i4')
        grp.copy(g['Name1'],'NOM')
        dset = f['{}/NOM'.format(grp.name)]
        dset[:] = ASCIIname(name)

        ElNum += num
        Cnct += connec

        NumEnd = NumSt + len(num)
        VArr[NumSt:NumEnd] = Elgrp

        Elgrp -= 1
        NumSt = NumEnd

    Tets = Elems.create_group('TE4')
    Tets.attrs.create('CGS',1,dtype='i4')
    Tets.attrs.create('CGT',1,dtype='i4')
    Tets.attrs.create('GEO',304,dtype='i4')
    Tets.attrs.create('PFL','MED_NO_PROFILE_INTERNAL',dtype='S100')

    Cnct = np.array(Cnct)

    NOD = Tets.create_dataset('NOD',data=np.hstack((Cnct[:,2],Cnct[:,1],Cnct[:,0],Cnct[:,3])), dtype='i4')
    NOD.attrs.create('CGT',1,dtype='i4')
    NOD.attrs.create('NBR',Nvols,dtype='i4')

    NUM = Tets.create_dataset('NUM',data=np.array(ElNum), dtype='i4')
    NUM.attrs.create('CGT',1,dtype='i4')
    NUM.attrs.create('NBR',Nvols,dtype='i4')

    FAM = Tets.create_dataset('FAM',data=VArr, dtype='i4')
    FAM.attrs.create('CGT',1,dtype='i4')
    FAM.attrs.create('NBR',Nvols,dtype='i4')

    # Surface groups if there are any
    if SGroupInfo:
        sorter = np.argsort(ElNum)

        SurfDict = GroupCross(SGroupInfo)

        SurfCount = max(ElNum)+1
        SurfNum, Surfgrp, SurfN1, SurfN2, SurfN3 = [], [], [], [], []
        IxDict = {1:np.array([0,1,2])}
        for key, val in SurfDict.items():
            for key1, val1 in val.items():
                tarr = np.array(val1)
                elem = tarr[:,0]
                fix = tarr[:,1]

                grp = ELEME.create_group('grp{}'.format(Elgrp))
                grp.attrs.create('NUM',Elgrp,dtype='i4')
                grp = grp.create_group('GRO')
                grp.attrs.create('NBR',key,dtype='i4')
                grp.copy(g['Name{}'.format(key)],'NOM')
                dset = f['{}/NOM'.format(grp.name)]
                dset[:] = ASCIIname(key1)

                Ixs = sorter[np.searchsorted(ElNum, elem, sorter=sorter)]
                for connect, faceix in zip(Cnct[Ixs],fix):
                    if faceix == 1: nds = connect[np.array([2,1,0])]
                    elif faceix == 2:nds = connect[np.array([0,1,3])]
                    elif faceix == 3:nds = connect[np.array([1,2,3])]
                    elif faceix == 4:nds = connect[np.array([2,0,3])]

                    SurfN1.append(nds[0]), SurfN2.append(nds[1]), SurfN3.append(nds[2])
                    SurfNum.append(SurfCount)
                    Surfgrp.append(Elgrp)
                    SurfCount += 1

                Elgrp -= 1

        #print(len(SurfNum),len(Surfgrp),len(SurfN1),len(SurfN2),len(SurfN3))
        Nsurf = len(SurfNum)
        Surf = Elems.create_group('TR3')
        Surf.attrs.create('CGS',1,dtype='i4')
        Surf.attrs.create('CGT',1,dtype='i4')
        Surf.attrs.create('GEO',203,dtype='i4')
        Surf.attrs.create('PFL','MED_NO_PROFILE_INTERNAL',dtype='S100')

        NOD = Surf.create_dataset('NOD',data=np.array(SurfN1 + SurfN2 + SurfN3), dtype='i4')
        NOD.attrs.create('CGT',1,dtype='i4')
        NOD.attrs.create('NBR',Nsurf,dtype='i4')

        NUM = Surf.create_dataset('NUM',data=np.array(SurfNum), dtype='i4')
        NUM.attrs.create('CGT',1,dtype='i4')
        NUM.attrs.create('NBR',Nsurf,dtype='i4')

        FAM = Surf.create_dataset('FAM',data=np.array(Surfgrp), dtype='i4')
        FAM.attrs.create('CGT',1,dtype='i4')
        FAM.attrs.create('NBR',Nsurf,dtype='i4')


    ### Nodes ###
    #f.copy(g['NOEUD'],'FAS/{}/NOEUD/'.format(MeshName))
    f.copy(g['ELEME'],'FAS/{}/NOEUD/'.format(MeshName))
    NOEUD = f['FAS/{}/NOEUD/'.format(MeshName)]
    NNodes = len(NdNum)
    NArr = np.zeros(NNodes,dtype='i4')

    # Add in node groups if there are any
    if NGroupInfo:
        NodeDict = GroupCross(NGroupInfo,0)

        Ncount = 1
        for key, val in NodeDict.items():
            for key1, val1 in val.items():
                grp = NOEUD.create_group('grp{}'.format(Ncount))
                grp.attrs.create('NUM',Ncount,dtype='i4')
                grp = grp.create_group('GRO')
                grp.attrs.create('NBR',key,dtype='i4')
                grp.copy(g['Name{}'.format(key)],'NOM')
                dset = f['{}/NOM'.format(grp.name)]
                dset[:] = ASCIIname(key1)
                NArr[np.array(val1)-1] = Ncount
                Ncount += 1

    Coords = np.array(Coords)
    COO = Nodes.create_dataset('COO',data=np.hstack((Coords[:,0],Coords[:,1],Coords[:,2])), dtype='f8')
    COO.attrs.create('CGT',1,dtype='i4')
    COO.attrs.create('NBR',NNodes,dtype='i4')
    NUM = Nodes.create_dataset('NUM',data=NdNum, dtype='i4')
    NUM.attrs.create('CGT',1,dtype='i4')
    NUM.attrs.create('NBR',NNodes,dtype='i4')
    FAM = Nodes.create_dataset('FAM', data=NArr, dtype='i4')
    FAM.attrs.create('CGT',1,dtype='i4')
    FAM.attrs.create('NBR',NNodes,dtype='i4')

    print("Mesh successfully converted and can be found at \n{}".format(fileout))


def ASCIIname(names):
    namelist = []
    for name in names:
        lis = [0]*80
        for i, char in enumerate(name):
            num = (ord(char))
            lis[i] = num

        a = np.array(lis)
        namelist.append(a)
    res = np.array(namelist)
    return res

def GroupCross(groups,check = None):
    Dict = {1 : groups.copy()}
    level = 1

    while any(list(Dict[level].values())):
        Dict[level+1] = {}

        for key, val in Dict[level].items():
            setval = set(val)

            for key1, val1 in Dict[1].items():
                if key1[0] in key:
                    continue

                CombName = tuple(sorted(key+key1))
                if CombName in Dict[level+1].keys():
                    continue

                intersection = list(setval.intersection(val1))
                Dict[level+1][CombName] = intersection

        if not any(list(Dict[level+1].values())):
            del Dict[level+1]
            break

        level+=1

    if level > 1:
        for i in range(2,level+1):
            levdict = Dict[i].copy()
            for key, val in levdict.items():
                if not val:
                    del Dict[i][key]
                    continue

                levdict1 = Dict[i-1].copy()
                for key1, val1 in levdict1.items():
                    if set(key).intersection(key1):
                        newdat = list(set(val1).difference(val))
                        if newdat:
                            Dict[i-1][key1] = newdat
                        else :
                            del Dict[i-1][key1]

    if check:
        # Original number in dictionary
        sum0, names0 = 0, []
        for key, val in groups.items():
            names0.append(key[0])
            sum0 += len(val)
        print('Original groups')
        print('Groups: {}'.format(sorted(names0)))
        print('Total number: {}'.format(sum0))
        print('')
        # Numbers in new layered dictionary
        sum1, names1 = 0, ()
        for key, val in Dict.items():
            for key1, val1 in val.items():
                names1 += key1
                sum1 += key*len(val1)
        print('Updated groups')
        print('Groups: {}'.format(sorted(names1)))
        print('Total number: {}'.format(sum1))
        print('')

    return Dict

if __name__=='__main__':
    mesh_path = sys.argv[1]
    main(mesh_path)