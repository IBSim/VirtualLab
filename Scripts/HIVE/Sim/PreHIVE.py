import shutil
import os
import sys
sys.dont_write_bytecode=True
import numpy as np
import h5py
import matplotlib.pyplot as plt
import time
from bisect import bisect_left as bl
import shutil

from Scripts.Common.VLFunctions import MeshInfo,ASCIIname
from Scripts.Common.VLPackages import SalomeRun, ERMESRun
from DA.Functions import Uniformity3 as UniformityScore


def HTC(VL, SimDict):
    '''This function calculates the heat flux between the fluid and pipe as a
    function of wall temperature. This data is used to apply a BC in the
    CodeAster simulation.'''

    CreateHTC = getattr(SimDict['Parameters'], 'CreateHTC', True)

    # if None then no HTC will be generated
    if CreateHTC == None: return

    HeatFlux = "{}/HeatTransfer.dat"
    if CreateHTC:
        # Create a new set of HTC values based on pipe geometries and coolant properties
        from CoolantHT.Coolant import Properties as ClProp
        from CoolantHT.Pipe import PipeGeom
        from CoolantHT.HeatFlux1D import HIVE_Coolant, Verify

        Pipedict = SimDict['Parameters'].Pipe
        Pipe = PipeGeom(shape=Pipedict['Type'], pipediameter=Pipedict['Diameter'], length=Pipedict['Length'])

        Cooldict = SimDict['Parameters'].Coolant
        Coolant = ClProp(T=Cooldict['Temperature']+273.15, P=Cooldict['Pressure'], velocity=Cooldict['Velocity'])

        # Check if properties of coolant are applicable for the correlations used.
        VerifyCorr = Verify(Coolant,Pipe,CorrFC='st', CorrSB='jaeri', CorrCHF='mt')

        # Get heat transfer data
        HTdata, HTdict = HIVE_Coolant([10,None,1], Coolant, Pipe, CorrFC='st', CorrSB='jaeri', CorrCHF='mt')

        np.savetxt(HeatFlux.format(SimDict['PREASTER']), HTdata, fmt = '%.2f %.8f')
        np.savetxt(HeatFlux.format(SimDict['TMP_CALC_DIR']), HTdata, fmt = '%.2f %.8f')

        import matplotlib.pyplot as plt
        plt.plot(HTdata[:,0],HTdata[:,1])
        plt.scatter(*HTdict['Saturation'],marker='x',label='Saturation\nTemperature')
        plt.scatter(*HTdict['ONB'],marker='x',label='Onset Nucleate\nBoiling')
        plt.scatter(*HTdict['CHF'],marker='x',label='Critical Heat\nFlux')
        plt.xlabel('Temperature C',fontsize=14)
        plt.ylabel('Heat Flux (W/m^2)',fontsize=14)
        plt.legend()
        plt.savefig("{}/HeatTransfer.png".format(SimDict['PREASTER']), bbox_inches='tight')
        plt.close()

        HTdict['VerifyFC'] = VerifyCorr[0]
        HTdict['VerifySB'] = VerifyCorr[1]
        HTdict['VerifyCHF'] = VerifyCorr[2]

        SimDict['Data']['HTdict'] = HTdict

    elif os.path.isfile(HeatFlux.format(SimDict['PREASTER'])):
        ### Use previous HTC values from PreAster directory
        shutil.copy(HeatFlux.format(SimDict['PREASTER']), SimDict['TMP_CALC_DIR'])
    else:
        ### Exit due to errors
        sys.exit("CreateHTC not 'True' and {} doesn't exist".format(HeatFlux.format(SimDict['PREASTER'])))

    SimDict['HTData'] = HeatFlux.format(SimDict['TMP_CALC_DIR'])

def ERMES_Mesh(VL, MeshIn, MeshOut, Parameters, tempdir='/tmp', AddPath=[], LogFile=None, GUI=0):
    '''
    MeshIn is used to build a conformal mesh (MeshOut) which is used by ERMES
    to generate the EM loads. A coil is added above the sample along with a
    vacuum which sits around both.
    '''

    script = "{}/EM/NewEM.py".format(VL.SIM_SCRIPTS)
    DataDict = {'Parameters':Parameters,'InputFile':MeshIn,'OutputFile':MeshOut}
    err = SalomeRun(script, DataDict=DataDict, AddPath=AddPath,
                    OutFile=LogFile, tempdir=tempdir, GUI=GUI)
    return err

def SetupERMES(VL, Parameters, ERMESMeshFile, tmpERMESdir, check=False):
    '''
    Creates the .dat files which are needed to perform an ERMES simulation.
    For the current version of ERMES (12.5) a static analysis is performed first,
    which generates the currents, followed by a full wave simulation.

    Check flag will provide additional output to the terminal which is verified
    during ERMES_Conversion.
    '''

    Temperatures = [20] ###TODO:sort this

    # Get mesh info using the MeshInfo class written using h5py
    ERMESMesh = MeshInfo(ERMESMeshFile)

    # Define duplicate nodes for contact surfaces, which is on the SampleSurface and CoilSurface
    CoilSurface = ERMESMesh.GroupInfo('CoilSurface')
    SampleSurface = ERMESMesh.GroupInfo('SampleSurface')
    ContactNodeSt = ERMESMesh.NbNodes + 1
    ContactNodes = SampleSurface.Nodes.tolist() + CoilSurface.Nodes.tolist()
    NbNodesERMES = ERMESMesh.NbNodes + len(ContactNodes)
#    ContactNodes = sorted(ContactNodes)
#    print(NbNodes, NbNodes+len(ContactNodes))

    ###### .dat file ######
    # Node part which will be in both Electrostatic and FullWave
    NodeList = list(range(1,ContactNodeSt)) + ContactNodes
    Coords = ERMESMesh.GetNodeXYZ(NodeList)
    strNodes = ["No[{}] = p({:.10f},{:.10f},{:.10f});\n".format(i+1,Crd[0],Crd[1],Crd[2]) for i,Crd in enumerate(Coords)]
    strNodes.insert(0,"// List of nodes\n")
    strNodes = "".join(strNodes)

    # Electrostatic File
    # Define problem type
    Stat01 = "// Setting problem\n" + \
    "ProblemType = Static;\n" + \
    "ProblemType = GiDTol9;\n" + \
    "ProblemType = RELSSOL;\n" + \
    "ProblemType = 1st;\n" + \
    "ProblemType = GAv;\n" + \
    "ProblemType = NRFIG;\n" + \
    "ProblemType = FSWRIF;\n" + \
    "ProblemType = OFFASCII;\n" + \
    "ProblemType = IMPJOFF;\n" + \
    "ProblemType = {}pr;\n".format(getattr(Parameters,'NbProc',1)) + \
    "ProblemType = LE;\n" + \
    "ProblemFrequency = {};\n".format(Parameters.Frequency*2*np.pi) + \
    "ProblemType = CheckConsistency;\n"

    # Define Material properties. Only need Electrical Conductivity of 1 for coil
    # for this analysis.
    EMlist = ['Vacuum','Coil'] + sorted(Parameters.Materials.keys())
    Electrolist = [0]*len(EMlist)
    Electrolist[1] = 1
    StatMat = "// Material properties\n"
    for i,res in enumerate(Electrolist):
        StatMat += "PROPERTIES[{}].IHL_ELECTRIC_CONDUCTIVITY  = {};\n".format(i+1,res) + \
        "PROPERTIES[{}].REAL_MAGNETIC_PERMEABILITY = {};\n".format(i+1,1) + \
        "PROPERTIES[{}].IMAG_MAGNETIC_PERMEABILITY = {};\n".format(i+1,0) + \
        "PROPERTIES[{}].REAL_ELECTRIC_PERMITTIVITY = {};\n".format(i+1,1) + \
        "PROPERTIES[{}].IMAG_ELECTRIC_PERMITTIVITY = {};\n".format(i+1,0)

    # Property used for CoilIn BC (100 is a nominal amount)
    StatMat += "// Special materials properties\n" + \
    "PROPERTIES[17].COMPLEX_IBC = [0.000000000000000000,100.000000000000000000];\n" + \
    "PROPERTIES[32].COMPLEX_IBC = [1.0,0.0];\n"

    # BC at CoilOut terminal
    StatBC =["No[{}].V.Fix(0.0);\n".format(nd) for nd in ERMESMesh.GroupInfo('CoilOut').Nodes]
    StatBC.insert(0,"// Fixing static voltage on nodes in nodes\n")
    StatBC = "".join(StatBC)

    # describes the building procedure for the problem
    Stat05 = "// Initializing building \n" + \
    "ElementsGroup = electromagnetic_group;\n\n" + \
    '// Generating debug results (if "Debug" mode activated) \n\n' + \
    "// Building and solving\n" + \
    "ProblemType = Build;\n"

    with open('{}/Static.dat'.format(tmpERMESdir),'w+') as f:
        f.write(Stat01 + strNodes + StatMat + StatBC + Stat05)

    # FullWave part
    Wave01 = "// Setting problem\n" + \
    "ProblemType = E3D;\n" + \
    "ProblemType = GiDTol9;\n" + \
    "ProblemType = RELSSOL;\n" + \
    "ProblemType = 1st;\n" + \
    "ProblemType = GAv;\n" + \
    "ProblemType = NRFIG;\n" + \
    "ProblemType = FSWRIF;\n" + \
    "ProblemType = OFFASCII;\n" + \
    "ProblemType = IMPJON;\n" + \
    "ProblemType = {}pr;\n".format(getattr(Parameters,'NbProc',1)) + \
    "ProblemType = LE;\n" + \
    "ProblemFrequency = {};\n".format(Parameters.Frequency*2*np.pi) + \
    "ProblemType = CheckConsistency;\n"

    WaveBC = ["// Creating High order nodes\n","ProblemType = CreateHONodes;\n","// Making contact elements\n", \
        "ProblemType = MakeContact;\n","// Fixing degrees of freedom in PEC nodes\n" ]

    PECEls = ERMESMesh.GroupInfo('VacuumSurface').Connect
    WaveBC += ["PEC = n([{},{},{}]);\n".format(Nodes[0],Nodes[1],Nodes[2]) for Nodes in PECEls]

    # _PECEls = ERMESMesh.GroupInfo('Void_0_Surfaces').Connect
    # WaveBC += ["PEC = n([{},{},{}]);\n".format(Nodes[0],Nodes[1],Nodes[2]) for Nodes in _PECEls]

    WaveBC = "".join(WaveBC)

    Wave05 = Stat05

    # Teperature dependent material properties for NL simulation
    for Temp in Temperatures:
        WaveMat = "// Material properties\n"
        for i, part in enumerate(EMlist):
            MgPrm, ElPrm = [1,0], [1,0]
            if part in ('Vacuum','Coil'):
                ElCnd = 0
            else :
                fpath = '{}/{}/{}.dat'.format(VL.MATERIAL_DIR,Parameters.Materials[part],'ElecCond')
                prop = np.fromfile(fpath,dtype=float,count=-1,sep=" ")
                ElCnd = np.interp(Temp,prop[::2],prop[1::2])

            WaveMat += "PROPERTIES[{}].IHL_ELECTRIC_CONDUCTIVITY  = {};\n".format(i+1,ElCnd) + \
            "PROPERTIES[{}].REAL_MAGNETIC_PERMEABILITY = {};\n".format(i+1,MgPrm[0]) + \
            "PROPERTIES[{}].IMAG_MAGNETIC_PERMEABILITY = {};\n".format(i+1,MgPrm[1]) + \
            "PROPERTIES[{}].REAL_ELECTRIC_PERMITTIVITY = {};\n".format(i+1,ElPrm[0]) + \
            "PROPERTIES[{}].IMAG_ELECTRIC_PERMITTIVITY = {};\n".format(i+1,ElPrm[1])
        WaveMat += "// Special materials properties\n" + \
        "PROPERTIES[17].COMPLEX_IBC = [0.000000000000000000,100.000000000000000000];\n" + \
        "PROPERTIES[32].COMPLEX_IBC = [1.0,0.0];\n"
        with open('{}/Wave{}.dat'.format(tmpERMESdir,Temp),'w+') as f:
            f.write(Wave01 + strNodes + WaveMat + WaveBC + Wave05)

    del strNodes, StatBC, WaveBC

    # Create variables for contact node information used in dat file 1 and 3
    # Replace node numbers with newly created nodes at same location
    Vacuumgrp = ERMESMesh.GroupInfo('Vacuum')
    VacuumNew = np.copy(Vacuumgrp.Connect)
    ContactFaceOrig = np.vstack((SampleSurface.Connect,CoilSurface.Connect))
    ContactFaceNew = np.copy(ContactFaceOrig)
    for i, nd in enumerate(ContactNodes):
        NewNode = ContactNodeSt+i
        VacuumNew[VacuumNew == nd] = NewNode
        ContactFaceNew[ContactFaceNew == nd] = NewNode

    ###### 1.dat file ######
    # Desribes the connectivity of the mesh. Same for Electrostatic and FullWave
    strMesh = ["// Volume elements\n"]
    for i,name in enumerate(EMlist):
        if i==0: GrpCnct = VacuumNew
        else: GrpCnct = ERMESMesh.GroupInfo(name).Connect
        for Nodes in GrpCnct:
            strMesh.append("VE({},{},{},{},{});\n".format(Nodes[2],Nodes[1],Nodes[0],Nodes[3],i+1))
    strMesh = "".join(strMesh)

    with open('{}/Static-1.dat'.format(tmpERMESdir),'w+') as f:
        f.write(strMesh)
    for Temp in Temperatures:
        with open('{}/Wave{}-1.dat'.format(tmpERMESdir,Temp),'w+') as f:
            f.write(strMesh)

    del strMesh

    ####### 2.dat file ######
    # BC for CoilIn terminal
    CoilInCnct = ERMESMesh.GroupInfo('CoilIn').Connect
    StatBC = ["GRC({},{},{},17);\n".format(Nodes[0],Nodes[1],Nodes[2]) for Nodes in CoilInCnct]
    StatBC.insert(0, "// Static Robin elements\n")
    StatBC = "".join(StatBC)
    with open('{}/Static-2.dat'.format(tmpERMESdir),'w+') as f:
        f.write(StatBC)


    ###### -2.dat & -3.dat file ######
    # Describes the contact elements. A prism is created using the old & newly created
    # nodes (volume is tecchnically 0)
    strContact = ["// Contact elements\n"]
    for OrigNd, NewNd in zip(ContactFaceOrig,ContactFaceNew):
        strContact.append("CE = n([{},{},{},{},{},{}]);\n".format(OrigNd[2],OrigNd[1],OrigNd[0],NewNd[2],NewNd[1],NewNd[0]))
    strContact = "".join(strContact)

    for Temp in Temperatures:
        with open('{}/Wave{}-3.dat'.format(tmpERMESdir, Temp),'w+') as f:
            f.write(strContact)
        # Need this blank file for Wave analysis to execute properly
        with open('{}/Wave{}-2.dat'.format(tmpERMESdir, Temp),'w+') as f:
            f.write("// Source elements\n")

    ##### -5.dat file ######
    # Describes what to solve for in each simulation
    # Electrostatis part
    Stat51 = "// Static solver\n" + \
    "LinearSolver Diagonal = Bi_Conjugate_Gradient(1000000,250,0.000000001000000);\n\n" + \
    "// Solving static problem\n" + \
    "ElectromagneticStrategy.Solve(electromagnetic_group);\n\n" + \
    "// Setting output files\n" + \
    "ProblemType = PrintHOMesh;\n\n"  + \
    "// Computing and printing J current density\n" + \
    "ProblemType = Show_J_Static_smoothed;\n\n" + \
    "// Computing and printing J current density\n" + \
    "ProblemType = Show_J_Static_GP;\n\n" + \
    "// Export currents to file\n" + \
    "ProblemType = Export_Static_Currents;\n\n"

    Stat52 = "// Print the results of the field integrals\n" + \
    "ProblemType = Project_Static_Fields;\n"

    with open('{}/Static-5.dat'.format(tmpERMESdir),'w+') as f:
        f.write(Stat51 + Stat52)

    # FullWave part
    Wave51 = "// Complex solver\n" + \
    "LinearSolver Diagonal = Bi_Conjugate_Gradient(1000000,250,0.000000001000000);\n\n" + \
    "// Solving\n" + \
    "ElectromagneticStrategy.Solve(electromagnetic_group);\n\n" + \
    "// Main results (E field)\n" + \
    "CalculateNodal(IMAG_E);\n" + \
    "CalculateNodal(REAL_E);\n" + \
    "CalculateNodal(MOD_E);\n\n" + \
    "// Derivatives (H field)\n" + \
    "ProblemType = CalculateH;\n" + \
    "CalculateNodal(IMAG_H);\n" + \
    "CalculateNodal(REAL_H);\n" + \
    "CalculateNodal(MOD_H);\n\n" + \
    "// J currents\n" + \
    "ProblemType = CalculateJ;\n\n"

    Wave52 = "// Projecting modes in port planes\n" + \
    "ProblemType = Project;\n\n" + \
    "// Printing new high order mesh\n" + \
    "ProblemType = PrintHOMesh;\n\n" + \
    "// Other results\n" + \
    "ProblemType = CalculateJouleHeating;\n" + \
    "// J currents\n" + \
    "Print(REAL_J);\n" + \
    "Print(IMAG_J);\n" + \
    "Print(MOD_J);\n" + \
    "// Printing main results (E field)\n" +\
    "Print(IMAG_E);\n" + \
    "Print(REAL_E);\n" + \
    "Print(MOD_E);\n" + \
    "// Printing derivatives (H field)\n" + \
    "Print(IMAG_H);\n" + \
    "Print(REAL_H);\n" + \
    "Print(MOD_H);\n\n"


    # if check is True then this is included to calculate the current in the
    # CoilIn terminal. This can then be verified against the answer calculated
    # from the results
    if check:
        if True:
            sample = ERMESMesh.GroupInfo('Sample')
            strSample = ["PVIE({},{},{},{},32);\n".format(*FNodes) for FNodes in sample.Connect]
            strSample.insert(0,"// // Field integration over a volume \n")
            strSample = "".join(strSample)
        else: strSample=""

        strFace = ["PSIE({},{},{},32);\n".format(FNodes[0],FNodes[1],FNodes[2]) for FNodes in CoilInCnct]
        strFace.insert(0,"// Field integration over a surface\n")
        strFace = "".join(strFace)

        strSample = strSample + "\n" + strFace

    else : strSample = ""

    for Temp in Temperatures:
        with open('{}/Wave{}-5.dat'.format(tmpERMESdir, Temp),'w+') as f:
            f.write(Wave51 + strSample + Wave52)

    ### -9.dat file
    # Output file where currents calculated in the electrostatic simulation is saved to
    name = '1'
    with open('{}/Static-9.dat'.format(tmpERMESdir),'w+') as f:
        f.write('{}\n0\n'.format(name))

    ERMESMesh.Close()


def ERMES_Conversion(VL, Parameters, ERMESMeshFile, ERMESResFile, tmpERMESdir, check=False):
    '''
    Takes the .post.res file generated by ERMES and writes in an rmed format
    which can be opened by ParaVis.
    Check flag verifies that additional information generated by ERMES is correct
    '''
    shutil.copy2(ERMESMeshFile,ERMESResFile)

    ERMESMesh = MeshInfo(ERMESMeshFile)

    # Take results from .post.res results file and create .rmed file to view in ParaVis
    # Todo -  a more efficient way of doign this without dictionary
    ResDict = {}
    Start, End = -1,-2
    with open('{}/{}{}.post.res'.format(tmpERMESdir,'Wave',20),'r') as f:
        for j,line in enumerate(f):
            split = line.split()
            if split[0] == 'Result':
                ResType = (split[1])[1:-1]
                Start = j+2
                End = j+1+ERMESMesh.NbNodes
                tmplist = []
                continue

            if Start <= j <= End:
                tmplist.append(list(map(float,split[1:])))
            elif j == End+1:
                ResDict[ResType] = np.array(tmplist)


    CoilInCnct = ERMESMesh.GroupInfo('CoilIn').Connect
    CoilInArea, CoilInCurr, CoilInCurrsq = 0, 0, 0
    for nodes in CoilInCnct:
        coor1, coor2, coor3 = ERMESMesh.GetNodeXYZ(nodes)
        J1, J2, J3 = ResDict['mod(J)'][nodes - 1]

        area = 0.5*np.linalg.norm(np.cross(coor2-coor1,coor3-coor1))
        CoilInArea += area
        CoilInCurr += area*(J1 + J2 + J3)/3
        CoilInCurrsq += area*(J1**2 + J2**2 + J3**2)/3

    ERMESMesh.Close()
    # Scaling factor to ensure that the current measured at the coil matches Sim.Current
    Scale = Parameters.Current/CoilInCurr

    if check:
        print('These values should match up with those on the output from ERMES:')
        print('Surface [1]: {:.6e} m^2'.format(CoilInArea))
        print('intSurf(|J|): {:.6e}'.format(*CoilInCurr))
        print('intSurf(|J|^2): {:.6e}\n'.format(*CoilInCurrsq))

    # Create rmed file with ERMES results
    ERMESrmed = h5py.File(ERMESResFile, 'a')
    # Some groups require specific formatting so take an empty group from format file
    Formats = h5py.File("{}/MED_Format.med".format(VL.COM_SCRIPTS),'r')

    GrpFormat = Formats['ELEME']
    for ResName, Result in ResDict.items():
        # Scale up results and update dictionary
        if ResName == 'Joule_heating':
            ResDict[ResName] = Result = Result*Scale**2
        else :
            ResDict[ResName] = Result = Result*Scale

        ERMESrmed.copy(GrpFormat,"CHA/{}".format(ResName))
        grp = ERMESrmed["CHA/{}".format(ResName)]
        grp.attrs.create('MAI','ERMES',dtype='S8')
        if Result.shape[1] == 1: NOM =  'Res'.ljust(16)
        elif Result.shape[1] == 3: NOM = 'DX'.ljust(16) + 'DY'.ljust(16) + 'DZ'.ljust(16)
        grp.attrs.create('NCO',Result.shape[1],dtype='i4')
        grp.attrs.create('NOM', NOM,dtype='S100')
        grp.attrs.create('TYP',6,dtype='i4')
        grp.attrs.create('UNI',''.ljust(len(NOM)),dtype='S100')
        grp.attrs.create('UNT','',dtype='S1')

        grp = grp.create_group('0000000000000000000100000000000000000001')
        grp.attrs.create('NDT',1,dtype='i4')
        grp.attrs.create('NOR',1,dtype='i4')
        grp.attrs.create('PDT',0.0,dtype='f8')
        grp.attrs.create('RDT',-1,dtype='i4')
        grp.attrs.create('ROR',-1,dtype='i4')
        grp = grp.create_group('NOE')
        grp.attrs.create('GAU','',dtype='S1')
        grp.attrs.create('PFL','MED_NO_PROFILE_INTERNAL',dtype='S100')
        grp = grp.create_group('MED_NO_PROFILE_INTERNAL')
        grp.attrs.create('GAU','',dtype='S1'    )
        grp.attrs.create('NBR', ERMESMesh.NbNodes, dtype='i4')
        grp.attrs.create('NGA',1,dtype='i4')
        grp.create_dataset("CO",data=Result.flatten(order='F'))

    ERMESrmed["CHA"].attrs.create('Current',Parameters.Current)
    ERMESrmed["CHA"].attrs.create('Frequency',Parameters.Frequency)

    ERMESrmed.close()
    Formats.close()

    return ResDict['Joule_heating']


def RunERMES(VL, Parameters, ERMESMeshFile, ERMESResFile, tmpERMESdir, check=False):
    '''
    This function runs the ERMES simulation, after creating the .dat files and
    then performs the conversion of the results to rmed format.
    '''
    # Since the coil is non-symmetric an electrostatic simulation is required before
    # the full wave simulation.
    Temperatures = [20] ### TODO: Sort out temperature part

    # Create ERMES .dat files
    SetupERMES(VL, Parameters, ERMESMeshFile, tmpERMESdir, check)

    # Run ERMES
    err = ERMESRun('{}/Static'.format(tmpERMESdir)) # Run static
    # Run Wave
    for Temp in Temperatures:
        err = ERMESRun('{}/Wave{}'.format(tmpERMESdir,Temp),Append=True)

    # Convert ERMES results file to rmed
    JH_Node = ERMES_Conversion(VL, Parameters, ERMESMeshFile, ERMESResFile, tmpERMESdir, check)

    # Copy LogFile
    shutil.copy("{}/ERMESLog".format(tmpERMESdir),os.path.dirname(ERMESResFile))

    # Calculate Joule heating for each volume element
    ERMESMesh = MeshInfo(ERMESMeshFile)
    Coor = ERMESMesh.GetNodeXYZ(list(range(1,ERMESMesh.NbNodes+1)))
    Sample = ERMESMesh.GroupInfo('Sample')
    JH_Vol, Volumes = [], []
    # tsts = np.zeros(Sample.NbNodes)
    for Nds in  Sample.Connect:
        v1,v2,v3,v4 = Coor[Nds-1]
        vol = 1/float(6)*abs(np.dot(np.cross(v2-v1,v3-v1),v4-v1))
        Volumes.append(vol)
        # tsts[Nds-1]+=vol/4

        # geometric average of 4 Joule heating values
        JH_avg = np.sum(JH_Node[Nds-1,:])/4
        JH_Vol.append(JH_avg)
    ERMESMesh.Close()

    # W = tsts*JH_Node.flatten()[:Sample.NbNodes]
    # print(W.sum())

    # Get sorted index (descending order) for JH_Vol & sort arrays by this
    # This is used for the thresholding capability
    JH_Vol = np.array(JH_Vol)
    sortlist = JH_Vol.argsort()[::-1]
    JH_Vol = JH_Vol[sortlist]
    Volumes = np.array(Volumes)[sortlist]
    Elements = Sample.Elements[sortlist]

    ERMESrmed = h5py.File(ERMESResFile, 'a')

    # Save arrays to ERMES.rmed file for easy access
    grp = ERMESrmed.create_group('EM_Load')
    grp.create_dataset('JH_Vol',data=JH_Vol)
    grp.create_dataset('Elements',data=Elements)
    grp.create_dataset('Volumes',data=Volumes)
    grp.create_dataset('JH_Node',data=JH_Node)
    grp.attrs.create('Current',Parameters.Current)
    grp.attrs.create('Frequency',Parameters.Frequency)
    grp.attrs.create('NbEls',JH_Vol.shape[0])

    ERMESrmed.close()

    return JH_Vol, Volumes, Elements, JH_Node

def ERMES(VL,MeshFile,ERMESresfile,Parameters,CalcDir,RunSim=True,GUI=False):
    '''
    Either get required results by running a new ERMES simulation or get results
    from results file.
    '''
    if RunSim:
        # Run ERMES simulation

        # Create ERMES mesh
        ERMESmesh = "{}/Mesh.med".format(CalcDir)
        err = ERMES_Mesh(VL, MeshFile, ERMESmesh, Parameters,
                         tempdir=CalcDir,AddPath=[VL.SIM_SCRIPTS], GUI=GUI)
        if err: return sys.exit('Issue creating mesh')

        # Run simulation using mesh created
        CheckERMES = getattr(Parameters,'CheckERMES',False)
        return RunERMES(VL, Parameters, ERMESmesh, ERMESresfile, CalcDir,CheckERMES)

    elif os.path.isfile(ERMESresfile):
        # Read in a previous set of ERMES results
        #======================================================================
        # Check results match
        ERMESres = h5py.File(ERMESresfile, 'r')
        attrs =  ERMESres["EM_Load"].attrs

        NbVolumes = MeshInfo(MeshFile, meshname='Sample').NbVolumes
        if attrs['NbEls'] != NbVolumes:
            sys.exit("ERMES.rmed file doesn't match with mesh used for {} simulation".format(Parameters.Name))
        if Parameters.Frequency != attrs['Frequency']:
            sys.exit("Frequencies do not match")

        #======================================================================
        # Get results from file and scale
        Scale = (Parameters.Current/attrs['Current'])**2
        Elements = ERMESres["EM_Load/Elements"][:]
        Volumes = ERMESres["EM_Load/Volumes"][:]
        JH_Vol = ERMESres["EM_Load/JH_Vol"][:]*Scale
        JH_Node =  ERMESres["EM_Load/JH_Node"][:]*Scale
        ERMESres.close()
        return JH_Vol, Volumes, Elements, JH_Node
    else :
        # exit due to error
        sys.exit("ERMES results file '{}' does not exist and RunERMES flag not set to True".format(ERMESresfile))

def EMI(VL, SimDict):
    '''
    This function gets the electromagnetic loads generated by the coil. To
    implement this BC in CodeAster a group is required for each element, which
    can be very time-consuming.    Additional options are available to 'compress'
    the data to speed up the simulation:
    Thresholding - Power generated by the coil impacts only a small percentage
                   of the elements. The thresholding fraction (0-1) finds the elements
                   which provide that percentage of the power, i.e. thresholding
                   value of 0.99 will find the elements which contribute 99% of the power.
    Clustering - Clusters the EM loads in to NbClusters number of groups. A
                 Goodness of Fit Value (GFV) specifies how well the data is represented
                 having been put in to clusters. A GFV score fo above 0.99 is recommended.
    Both methods can be applied together, however generally clustering is the advised
    method as this guarantees a fixed number of mesh groups to be created.
    '''

    Parameters = SimDict['Parameters']
    # Get EM loads, either read in from file or created from ERMES simulation
    ERMESresfile = '{}/ERMES.rmed'.format(SimDict['PREASTER'])
    ERMESdir = "{}/ERMES".format(SimDict['TMP_CALC_DIR'])
    os.makedirs(ERMESdir)
    RunERMES = getattr(Parameters,'RunERMES',True)
    EM_GUI = getattr(Parameters,'EM_GUI',False)
    JH_Vol, Volumes, Elements, JH_Node = ERMES(VL,SimDict['MeshFile'],
                                        ERMESresfile,Parameters,
                                        ERMESdir,RunERMES, GUI=EM_GUI)
    shutil.rmtree(ERMESdir) #rm ERMES dir here as this can be quite large

    Watts = JH_Vol*Volumes

    CoilPower = Watts.sum()
    print("Power delivered by coil: {:.4f}W".format(CoilPower))
    SimDict['CoilPower'] = CoilPower

    # Uniformity = UniformityScore(JH_Node,ERMESresfile)
    # SimDict['Uniformity'] = Uniformity

    Threshold = getattr(Parameters,'Threshold', 0)
    if not Threshold:
        pass
    elif Threshold<1 and Threshold>0:
        CumSum = Watts.cumsum()/CoilPower
        # Find position in CumSum where the threshold percentage has been reached
        pos = bl(CumSum,Threshold)
        JH_Vol = JH_Vol[:pos+1]
        Volumes = Volumes[:pos+1]
        Elements = Elements[:pos+1]
        Watts = Watts[:pos+1]

        print("The {} most influential elements will be assigned EM loads ({}% threshold)".format(pos+1, Threshold*100))
        if getattr(SimDict['Parameters'],'EMScale', True):
            # If EMScale is True then the power input will be scaled to the original value.
            # i.e. if EMThreshold is 0.5 then the magnitude for each element will be doubled.
            sc = 1/CumSum[pos]
            Watts,JH_Vol = Watts*sc,JH_Vol*sc

        if getattr(Parameters,'ThresholdPlot', True):
            NbEls = CumSum.shape[0]
            Percentages = [0.9,0.99,0.999,0.9999]
            xlog = np.log10(np.arange(1,NbEls+1))
            x = xlog/xlog[-1]
            fig = plt.figure(figsize = (10,8))
            plt.plot(x, CumSum, label="Cumulative power")
            ticks, labels = [0], [0]
            for prc in Percentages:
                pos = bl(CumSum,prc)
                num = np.log10(pos+1)/xlog[-1]
                plt.plot([num, num], [0, prc], '--',label="{}% of power ({} Elements)".format(prc*100,pos+1))
                ticks.append(num)
                labels.append(round((pos+1)/NbEls,3))

            plt.plot([1, 1], [0, 1], '--',label="100% of power ({} Elements)".format(NbEls))
            ticks.append(1)
            labels.append(1)
            plt.xticks(ticks, labels,rotation="vertical")
            plt.legend(loc='upper left')
            plt.xlabel('Fraction of total elements required')
            plt.ylabel('Power (scaled)')
            plt.savefig("{}/Thresholding.png".format(SimDict['PREASTER']))
            plt.close()
    else:
        print("All ({}) elements will be assigned EM loads".format(Elements.shape[0]))

    NbClusters = int(getattr(Parameters,'NbClusters',0))
    if NbClusters > 0 and NbClusters < JH_Vol.shape[0]:
        '''
        Using K-means algorithm the loads will be grouped in to clusters, which
        will substantially speed up the simulation in CodeAster.
        The Goodness of Fit Value (GFV) describes how well the clustering
        represents the data, ranging from 0 (worst) to 1 (best).
        '''
        np.random.seed(123)
        from sklearn.cluster import KMeans
        X = JH_Vol.reshape(-1,1)
        X_sc = (X - X.min())/(X.max()-X.min())
        kmeans = KMeans(n_clusters=NbClusters).fit(X_sc)

        # Goodness of Fit Value is a metric of how good the clustering is
        SDAM = ((X_sc - X_sc.mean())**2).sum() # Squared deviation for mean array
        SDCM = kmeans.inertia_ # Squared deviation class mean
        GFV = (SDAM-SDCM)/SDAM # Goodness of fit value
        print("The {} elements are clustered in to {} groups.\n"\
              "Goodness of Fit Value: {}".format(Elements.shape[0],NbClusters,GFV))

        EM_Groups = [Elements[kmeans.labels_==i] for i in range(NbClusters)]
        EM_Loads = kmeans.cluster_centers_*(X.max()-X.min()) + X.min()

        # Scale cluster centres to ensure anticipated power is delivered
        # This is usually a very minor adjustment
        Watts_cl = EM_Loads[kmeans.labels_].flatten()*Volumes
        EM_Loads = EM_Loads*(Watts.sum()/Watts_cl.sum())
    else:
        EM_Groups = Elements
        EM_Loads = JH_Vol


    # Create mesh groups based on elements in EM_Groups
    tmpMeshFile = "{}/Mesh.med".format(SimDict["TMP_CALC_DIR"])
    GroupBy = getattr(Parameters,'GroupBy','H5PY')
    # This routine uses h5py to create the groups. It is the fastest available method.
    if GroupBy == 'H5PY':
        # Copy MeshFile to add groups to
        shutil.copy(SimDict['MeshFile'],tmpMeshFile)

        # Open MeshFile using h5py to append groups to
        tmpMeshMed = h5py.File(tmpMeshFile,'a')
        ElInfo = tmpMeshMed["ENS_MAA/Sample/-0000000000000000001-0000000000000000001/MAI/TE4"]
        ElList = ElInfo["NUM"][:]
        ElFam = ElInfo["FAM"][:]
        # Elbool = np.searchsorted(ElList,Elements)
        # ElList = ElList[Elbool]
        # ElFam = ElFam[Elbool]

        # Get minimum group famiy Id number (to avoid overwriting) and create dictionary containing
        # the affected groups
        ElGrps = tmpMeshMed["FAS/Sample/ELEME"]
        MinNum, GrpName = 0, {}
        for grpname in ElGrps.keys():
            grpnum = ElGrps[grpname].attrs['NUM']
            MinNum = min(MinNum,grpnum)
            if grpnum in ElFam:
                GrpName[grpnum] = ElGrps[grpname]
        NewNum = MinNum-1 # Number which groping will start at
        # Formats file contains format needed to add group
        Formats = h5py.File("{}/MED_Format.med".format(VL.COM_SCRIPTS),'r')
        for i,els in enumerate(EM_Groups):
            ElIxcl = np.searchsorted(ElList,els) # get indices of els in ElList
            ElFamcl = ElFam[ElIxcl] # get Family Ids associated with els
            for fam in np.unique(ElFamcl):
                NameGrps = GrpName[fam]['GRO/NOM'][:] # group names already associated with this family id
                EMnames = ASCIIname(['_EMgrp','_{}'.format(i)]) #ASCII name for 2 new groups; _EMgrp and M#
                NumGrps = NameGrps.shape[0]+2 # add 2 to NumGrps since we're creating 2 new groups
                dsetFormat = Formats["Name{}".format(NumGrps)] # copy group format from Formats

                # Create new group containing relevant information
                h5grp = "Grp{}".format(NewNum)
                ElGrps.copy(dsetFormat,"{}/GRO/NOM".format(h5grp))
                ElGrps[h5grp].attrs.create('NUM',NewNum,dtype='i4')
                ElGrps["{}/GRO".format(h5grp)].attrs.create('NBR',NumGrps,dtype='i4')
                ElGrps["{}/GRO/NOM".format(h5grp)][:] = np.vstack((NameGrps,EMnames))

                # Update ElFam with new family IDs created
                IxChange = ElIxcl[ElFamcl==fam]
                ElFam[IxChange] = NewNum

                NewNum -=1

        Formats.close()

        ElInfo["FAM"][:] = ElFam
        tmpMeshMed.close()
    # elif GroupBy == 'SALOME':
    #     ### May be broken so not reliable
    #     ArgDict = {"MeshFile":SimDict["MeshFile"], "tmpMesh":tmpMeshFile,"EMLoadFile":EMLoadFile}
    #     EMGroupFile = "{}/CreateEMGroups.py".format(os.path.dirname(os.path.abspath(__file__)))
    #     VL.SalomeRun(EMGroupFile, ArgDict=ArgDict)

    # Change MeshFile to point to mesh file created in TMP_CALC_DIR containing groups for EM load
    SimDict['MeshFile'] = tmpMeshFile

    # Write results to tmp file for Code_Aster to read in
    EMLoadFile = '{}/ERMES.npy'.format(SimDict['TMP_CALC_DIR'])
    np.save(EMLoadFile, EM_Loads)

def Single(VL, SimDict):
    HTC(VL, SimDict)

    if SimDict['Parameters'].EMLoad == 'ERMES':
        EMI(VL, SimDict)
