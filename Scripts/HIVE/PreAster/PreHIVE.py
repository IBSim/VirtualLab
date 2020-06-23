import shutil
import os
import sys
sys.dont_write_bytecode=True
import numpy as np
import h5py
from subprocess import Popen
from VLFunctions import MeshInfo
import matplotlib.pyplot as plt
import time
from bisect import bisect_left as bl

def GetHTC(Info, StudyDict):
#	for Name, StudyDict in Info.Studies.items():
	CreateHTC = getattr(StudyDict['Parameters'], 'CreateHTC', True)

	if CreateHTC == None: return
	# Create a new set of HTC values
	if CreateHTC:
		from HTC.Coolant import Properties as ClProp
		from HTC.Pipe import PipeGeom
		from HTC.ITER import htc as htc_ITER
		from HTC.berglesrohsenow import get_T_onb

		Pipedict = StudyDict['Parameters'].Pipe
		Pipe = PipeGeom(shape=Pipedict['Type'], pipediameter=Pipedict['Diameter'], length=Pipedict['Length'])


		Cooldict = StudyDict['Parameters'].Coolant
		Coolant = ClProp(T=Cooldict['Temperature']+273, P=Cooldict['Pressure'], velocity=Cooldict['Velocity'])

		# for key, val in Coolant.__dict__.items():
		# 	print(key, val)

		# Onset of Nucleat boiling
		T_onb = get_T_onb(Coolant,Pipe)

		# Starting WallTemp and increment between temperatures to check
		WallTemp, incr = 5, 5
		HTC = []
		while True:
			h = htc_ITER(Coolant, Pipe, WallTemp+273,T_onb=T_onb)
			# if h == 0: break
			if WallTemp > 200: break
			HTC.append([WallTemp, h])
			WallTemp += incr

		HTC = np.array(HTC)
		np.savetxt("{}/HTC.dat".format(StudyDict['PREASTER']), HTC, fmt = '%.2f %.8f')
		np.savetxt("{}/HTC.dat".format(StudyDict['TMP_CALC_DIR']), HTC, fmt = '%.2f %.8f')

		import matplotlib.pyplot as plt
		plt.plot(HTC[:,0],HTC[:,1])
		plt.savefig("{}/PipeHTC.png".format(StudyDict['PREASTER']), bbox_inches='tight')
		plt.close()

	### Use previous HTC values
	elif os.path.isfile("{}/HTC.dat".format(StudyDict['PREASTER'])):
		shutil.copy("{}/HTC.dat".format(StudyDict['PREASTER']), StudyDict['TMP_CALC_DIR'])

	### Exit due to errors
	else: Info.Exit("CreateHTC not 'True' and {} contains no HTC.dat file".format(StudyDict['PREASTER']))


#def ErmesRun(Info,study,NNodes):

##	if Info.Studies[study]['Parameters'].EMtype == 'loop':
##		for ermfile in RunFiles:
##			ERMES_run = Popen('ERMESv12.0 {}/{}/{}'.format(Info.TMP_DIR,study,ermfile), shell = 'TRUE')
##			ERMES_run.wait()
##	else :

#	Erstr= ''
#	for ERMinf in Info.Studies[study]['ERMES']:
#		Erstr = Erstr + 'ERMESv12.5 {}/{}/{} & '.format(Info.TMP_DIR,study,ERMinf[1])

#	ERMES_run = Popen(Erstr[:-2], shell = 'TRUE')
#	ERMES_run.wait()
#	Info.CheckProc(ERMES_run)

#	Results = []
#	for ERMinf in Info.Studies[study]['ERMES']:
#		ERMres = [ERMinf[0]]
#		with open('{}/{}/{}.post.res'.format(Info.TMP_DIR,study,ERMinf[1]),'r') as f:
#			for j,line in enumerate(f):
#				if j < 3 :
#					continue
#				elif j >= 3+NNodes :
#					break
#				else :
#					ERMres.append(float(line.split()[1]))

#		Results.append(ERMres)

#	res = (np.array(Results)).transpose()

#	np.savetxt('{}/{}/EM.dat'.format(Info.TMP_DIR,study),res,fmt = '%.8f',delimiter = '   ')
#	np.savetxt('{}/EM.dat'.format(Info.Studies[study]['DATA_DIR']),res,fmt = '%.8f',delimiter = '   ')

#	### Remove a file which gets created in the PWD
#	try:
#		os.remove('vector_Xo.dat')
#	except :
#		pass

def CoilCurrent(EMMesh, JRes, groupname = 'CoilIn', **kwargs):
	facesum, intJ, intJsq = 0, 0, 0
	JRes = np.array(JRes)
	for nodes in EMMesh.GroupInfo(groupname).Connect:
		coor1, coor2, coor3 = EMMesh.GetNodeXYZ(nodes)
		area = 0.5*np.linalg.norm(np.cross(coor2-coor1,coor3-coor1))
		facesum += area

		J1, J2, J3 = JRes[nodes - 1]
		intJ += area*(J1 + J2 + J3)/3
		intJsq += area*(J1**2 + J2**2 + J3**2)/3

	if 'verbosity' in kwargs.keys():
		if kwargs['verbosity'] in (True,'True'):
			print('These values should match up with those on the output from ERMES')
			print('Area: {:.6e}'.format(facesum))
			print('intSurf|J|: {:.6e}'.format(intJ))
			print('intSurf|J|^2: {:.6e}'.format(intJsq))

	return intJ

def SetupERMES(Info, StudyDict, ERMESout, **kwargs):
	check = kwargs.get('check', False)

	Temperatures = [20]
	MeshFile = "{}/{}.med".format(Info.MESH_DIR,StudyDict['Parameters'].Mesh)
	ERMESMesh = MeshInfo(MeshFile, meshname='xERMES')
	CoilSurface = ERMESMesh.GroupInfo('CoilSurface')
	SampleSurface = ERMESMesh.GroupInfo('SampleSurface')

	# Define duplicate nodes for contact surfaces, which is on the SampleSurface and Coil Surface
	ContactNodeSt = ERMESMesh.NbNodes + 1
	ContactNodes = SampleSurface.Nodes.tolist() + CoilSurface.Nodes.tolist()
	NbNodesERMES = ERMESMesh.NbNodes + len(ContactNodes)
#	ContactNodes = sorted(ContactNodes)
#	print(NbNodes, NbNodes+len(ContactNodes))

	###### .dat file ######
	# Node part which will be in both Electrostatic and FullWave
	NodeList = list(range(1,ContactNodeSt)) + ContactNodes
	Coords = ERMESMesh.GetNodeXYZ(NodeList)
	strNodes = ["No[{}] = p({:.10f},{:.10f},{:.10f});\n".format(i+1,Crd[0],Crd[1],Crd[2]) for i,Crd in enumerate(Coords)]
	strNodes.insert(0,"// List of nodes\n")
	strNodes = "".join(strNodes)

	# Electrostatic part
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
	"ProblemType = 2pr;\n" + \
	"ProblemType = LE;\n" + \
	"ProblemFrequency = {};\n".format(StudyDict['Parameters'].Frequency*2*np.pi) + \
	"ProblemType = CheckConsistency;\n"

	EMlist = ['Vacuum','Coil'] + sorted(StudyDict['Parameters'].Materials.keys())
	# Define Electric Conductivity for electrostatic part - all 0 except the Coil
	Electrolist = [0]*len(EMlist)
	Electrolist[1] = 1

	StatMat = "// Material properties\n"
	for i,res in enumerate(Electrolist):
		StatMat += "PROPERTIES[{}].IHL_ELECTRIC_CONDUCTIVITY  = {};\n".format(i+1,res) + \
		"PROPERTIES[{}].REAL_MAGNETIC_PERMEABILITY = {};\n".format(i+1,1) + \
		"PROPERTIES[{}].IMAG_MAGNETIC_PERMEABILITY = {};\n".format(i+1,0) + \
		"PROPERTIES[{}].REAL_ELECTRIC_PERMITTIVITY = {};\n".format(i+1,1) + \
		"PROPERTIES[{}].IMAG_ELECTRIC_PERMITTIVITY = {};\n".format(i+1,0)
	StatMat += "// Special materials properties\n" + \
	"PROPERTIES[17].COMPLEX_IBC = [0.000000000000000000,100.000000000000000000];\n" + \
	"PROPERTIES[32].COMPLEX_IBC = [1.0,0.0];\n"

	StatBC =["No[{}].V.Fix(0.0);\n".format(nd) for nd in ERMESMesh.GroupInfo('CoilOut').Nodes]
	StatBC.insert(0,"// Fixing static voltage on nodes in nodes\n")
	StatBC = "".join(StatBC)

	Stat05 = "// Initializing building \n" + \
	"ElementsGroup = electromagnetic_group;\n\n" + \
	'// Generating debug results (if "Debug" mode activated) \n\n' + \
	"// Building and solving\n" + \
	"ProblemType = Build;\n"

	with open('{}/Static.dat'.format(StudyDict['TMP_CALC_DIR']),'w+') as f:
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
	"ProblemType = 2pr;\n" + \
	"ProblemType = LE;\n" + \
	"ProblemFrequency = {};\n".format(StudyDict['Parameters'].Frequency*2*np.pi) + \
	"ProblemType = CheckConsistency;\n"

	WaveBC = ["// Creating High order nodes\n","ProblemType = CreateHONodes;\n","// Making contact elements\n", \
		"ProblemType = MakeContact;\n","// Fixing degrees of freedom in PEC nodes\n" ]

	PECEls = ERMESMesh.GroupInfo('VacuumSurface').Connect
	WaveBC += ["PEC = n([{},{},{}]);\n".format(Nodes[0],Nodes[1],Nodes[2]) for Nodes in PECEls]
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
				fpath = '{}/{}/{}.dat'.format(Info.MATERIAL_DIR,StudyDict['Parameters'].Materials[part],'ElecCond')
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

		with open('{}/Wave{}.dat'.format(StudyDict['TMP_CALC_DIR'],Temp),'w+') as f:
			f.write(Wave01 + strNodes + WaveMat + WaveBC + Wave05)

	del strNodes, StatBC, WaveBC

	# Create variables for contact node information used in dat file 1 and 3
	Vacuumgrp = ERMESMesh.GroupInfo('Vacuum')
	VacuumNew = np.copy(Vacuumgrp.Connect)
	ContactFaceOrig = np.vstack((SampleSurface.Connect,CoilSurface.Connect))
	ContactFaceNew = np.copy(ContactFaceOrig)
	for i, nd in enumerate(ContactNodes):
		NewNode = ContactNodeSt+i
		VacuumNew[VacuumNew == nd] = NewNode
		ContactFaceNew[ContactFaceNew == nd] = NewNode

	###### 1.dat file ######
	strMesh = ["// Volume elements\n"]
	for i,name in enumerate(EMlist):
		if i==0: GrpCnct = VacuumNew
		else: GrpCnct = ERMESMesh.GroupInfo(name).Connect
		for Nodes in GrpCnct:
			strMesh.append("VE({},{},{},{},{});\n".format(Nodes[2],Nodes[1],Nodes[0],Nodes[3],i+1))
	strMesh = "".join(strMesh)

	with open('{}/Static-1.dat'.format(StudyDict['TMP_CALC_DIR']),'w+') as f:
		f.write(strMesh)
	for Temp in Temperatures:
		with open('{}/Wave{}-1.dat'.format(StudyDict['TMP_CALC_DIR'],Temp),'w+') as f:
			f.write(strMesh)

	del strMesh

	####### 2.dat file ######
	CoilInCnct = ERMESMesh.GroupInfo('CoilIn').Connect
	StatBC = ["GRC({},{},{},17);\n".format(Nodes[0],Nodes[1],Nodes[2]) for Nodes in CoilInCnct]
	StatBC.insert(0, "// Static Robin elements\n")
	StatBC = "".join(StatBC)
	with open('{}/Static-2.dat'.format(StudyDict['TMP_CALC_DIR']),'w+') as f:
		f.write(StatBC)


	###### -3.dat file ######
	strContact = ["// Contact elements\n"]
	for OrigNd, NewNd in zip(ContactFaceOrig,ContactFaceNew):
		strContact.append("CE = n([{},{},{},{},{},{}]);\n".format(OrigNd[2],OrigNd[1],OrigNd[0],NewNd[2],NewNd[1],NewNd[0]))
	strContact = "".join(strContact)

	for Temp in Temperatures:
		with open('{}/Wave{}-3.dat'.format(StudyDict['TMP_CALC_DIR'], Temp),'w+') as f:
			f.write(strContact)
		# Need this blank file for Wave analysis to execute properly
		with open('{}/Wave{}-2.dat'.format(StudyDict['TMP_CALC_DIR'], Temp),'w+') as f:
			f.write("// Source elements\n")

	##### -5.dat file ######
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

	with open('{}/Static-5.dat'.format(StudyDict['TMP_CALC_DIR']),'w+') as f:
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
	"Print(MOD_J);\n\n"

	if check:
		strFace = ["PSIE({},{},{},32);\n".format(FNodes[0],FNodes[1],FNodes[2]) for FNodes in CoilInCnct]
		strFace.insert(0,"// Field integration over a surface\n")
		strFace = "".join(strFace)
	else : strFace = ""

	ERMESwave = []
	for Temp in Temperatures:
		with open('{}/Wave{}-5.dat'.format(StudyDict['TMP_CALC_DIR'], Temp),'w+') as f:
			f.write(Wave51 + strFace + Wave52)

		ERMESwave.append("ERMESv12.5 Wave{};".format(Temp))

	### -9.dat file
	name = '1'
	with open('{}/Static-9.dat'.format(StudyDict['TMP_CALC_DIR']),'w+') as f:
		f.write('{}\n0\n'.format(name))

	Ermesstr = "cd {}; ERMESv12.5 {};{}".format(StudyDict['TMP_CALC_DIR'],'Static',''.join(ERMESwave))
	ERMES_run = Popen(Ermesstr, shell = 'TRUE')
	ERMES_run.wait()

	ResDict = {}
	Start, End = -1,-2
	with open('{}/{}{}.post.res'.format(StudyDict['TMP_CALC_DIR'],'Wave',20),'r') as f:
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

	Jres = ResDict['mod(J)']
	CoilInArea, CoilInCurr, CoilInCurrsq = 0, 0, 0
	for nodes in CoilInCnct:
		coor1, coor2, coor3 = ERMESMesh.GetNodeXYZ(nodes)
		J1, J2, J3 = Jres[nodes - 1]

		area = 0.5*np.linalg.norm(np.cross(coor2-coor1,coor3-coor1))
		CoilInArea += area
		CoilInCurr += area*(J1 + J2 + J3)/3
		CoilInCurrsq += area*(J1**2 + J2**2 + J3**2)/3

	if check:
		print('These values should match up with those on the output from ERMES')
		print('Area: {:.6e}'.format(CoilInArea))
		print('intSurf|J|: {:.6e}'.format(CoilInCurr))
		print('intSurf|J|^2: {:.6e}'.format(CoilInCurrsq))

	# Create rmed file with ERMES results
	ERMESrmed = h5py.File(ERMESout, 'w')

	# Copy Mesh data from mesh .med file
	MeshMed = h5py.File(StudyDict['MeshFile'], 'r')
	ERMESrmed.copy(MeshMed["INFOS_GENERALES"],"INFOS_GENERALES")
	ERMESrmed.copy(MeshMed["FAS/xERMES"],"FAS/xERMES")
	ERMESrmed.copy(MeshMed["ENS_MAA/xERMES"],"ENS_MAA/xERMES")
	MeshMed.close()

	# Some groups require specific formatting so take an empty group from format file
	Formats = h5py.File("{}/MED_Format.med".format(Info.COM_SCRIPTS),'r')
	GrpFormat = Formats['ELEME']
	for ResName, Result in ResDict.items():
		ERMESrmed.copy(GrpFormat,"CHA/{}".format(ResName))
		grp = ERMESrmed["CHA/{}".format(ResName)]
		grp.attrs.create('MAI','xERMES',dtype='S8')
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
		grp.attrs.create('GAU','',dtype='S1'	)
		grp.attrs.create('NBR', ERMESMesh.NbNodes, dtype='i4')
		grp.attrs.create('NGA',1,dtype='i4')
		grp.create_dataset("CO",data=Result.flatten(order='F'))

	JH = ResDict["Joule_heating"]
	Coor = ERMESMesh.GetNodeXYZ(list(range(1,ERMESMesh.NbNodes+1)))
	# Volumetric results
	ERMESrmed.copy(GrpFormat,"GAUSS")
	grp = ERMESrmed["GAUSS"]
	grp = grp.create_group('TE4_____FPG1')
	grp.attrs.create('DIM',3,dtype='i4')
	grp.attrs.create('GEO', 304, dtype='i4')
	grp.attrs.create('INM', '', dtype='S100')
	grp.attrs.create('NBR', 1, dtype='i4')
	grp.create_dataset('COO',data=[0,0,0,1,1,0,0,0,0,0,1,0],dtype='f8')
	grp.create_dataset('GAU',data=[0.25]*3,dtype='f8')
	grp.create_dataset('VAL', data=[1/6],dtype='f8')

	Sample = ERMESMesh.GroupInfo('Sample')
	WattsPV, Watts, Volume = [], [], []
	for Nds in  Sample.Connect:
		VCoor = Coor[Nds-1]
		vol = 1/float(6)*abs(np.dot(np.cross(VCoor[1,:]-VCoor[0,:],VCoor[2,:]-VCoor[0,:]),VCoor[3,:]-VCoor[0,:]))
		# geometric average of nodal JH to element values
		Elsum = np.sum(JH[Nds-1,:])/4
		Volume.append(vol)
		WattsPV.append(Elsum)
		Watts.append(vol*Elsum)

	FWattsPV = np.zeros(ERMESMesh.NbVolumes)
	FWattsPV[:len(WattsPV)] = WattsPV
	FWatts = np.zeros(ERMESMesh.NbVolumes)
	FWatts[:len(Watts)] = Watts

	for Name, Values in zip(['WattsPV', 'Watts'],[FWattsPV, FWatts]):
		ERMESrmed.copy(GrpFormat,"CHA/{}".format(Name))
		grp = ERMESrmed["CHA/{}".format(Name)]
		grp.attrs.create('MAI','xERMES',dtype='S8')
		grp.attrs.create('NCO',1,dtype='i4')
		grp.attrs.create('NOM', 'Res'.ljust(16),dtype='S100')
		grp.attrs.create('TYP',6,dtype='i4')
		grp.attrs.create('UNI',''.ljust(16),dtype='S100')
		grp.attrs.create('UNT','',dtype='S1')

		grp = grp.create_group('0000000000000000000100000000000000000001')
		grp.attrs.create('NDT',1,dtype='i4')
		grp.attrs.create('NOR',1,dtype='i4')
		grp.attrs.create('PDT',0.0,dtype='f8')
		grp.attrs.create('RDT',-1,dtype='i4')
		grp.attrs.create('ROR',-1,dtype='i4')
		grp = grp.create_group('MAI.TE4')
		grp.attrs.create('GAU','TE4_____FPG1',dtype='S100')
		grp.attrs.create('PFL','MED_NO_PROFILE_INTERNAL',dtype='S100')
		grp = grp.create_group('MED_NO_PROFILE_INTERNAL')
		grp.attrs.create('GAU','TE4_____FPG1',dtype='S100'	)
		grp.attrs.create('NBR', ERMESMesh.NbVolumes, dtype='i4')
		grp.attrs.create('NGA',1,dtype='i4')
		grp.create_dataset("CO",data=Values)

	grp = ERMESrmed.create_group('EM_Load')
	# Get sorting index in descending order for WattsPV
	WattsPV = np.array(WattsPV)
	sortlist = WattsPV.argsort()[::-1]
	# Sort by sortlist for thresholding capabilities
	WattsPV = WattsPV[sortlist]*(1/CoilInCurr)**2
	Watts = np.array(Watts)[sortlist]*(1/CoilInCurr)**2
	Elements = Sample.Elements[sortlist]
	# Save arrays to ERMES.rmed file for easy access
	grp.create_dataset('WattsPV',data=WattsPV)
	grp.create_dataset('Watts',data=Watts)
	grp.create_dataset('Elements',data=Elements)

	# Thresholding image
	CumSum = Watts.cumsum()
	NbEls = CumSum.shape[0]
	CumSum = CumSum/CumSum[-1]
	Percentages = [0.5,0.9,0.99,0.999,0.9999]

	fig = plt.figure(figsize = (10,8))
	xlog = np.log10(np.arange(1,NbEls+1))
	xmax = xlog[-1]
	x = xlog/xmax
	plt.plot(x, CumSum, label="Cumulative power")
	ticks, labels = [0], [0]
	for prc in Percentages:
		pos = bl(CumSum,prc)
		num = np.log10(pos+1)/xmax
		plt.plot([num, num], [0, prc], '--',label="{}% of power ({} Elements)".format(prc*100,pos+1))
		frac = round((pos+1)/NbEls,3)
		ticks.append(num)
		labels.append(frac)
		# print("For {}% of the coil power, you will need {} elements ({}% total elements)".format(prc*100,pos+1,round(frac*100,2)))
	plt.plot([1, 1], [0, 1], '--',label="100% of power ({} Elements)".format(NbEls))
	ticks.append(1)
	labels.append(1)
	plt.xticks(ticks, labels,rotation="vertical")
	plt.legend(loc='upper left')
	plt.xlabel('Fraction of total elements required')
	plt.ylabel('Power (scaled)')
	plt.savefig("{}/EM_Thresholding".format(StudyDict['PREASTER']))
	plt.close()

	# fig = plt.figure(figsize = (14,5))
	# x = np.linspace(1/NbEls,1,NbEls)
	# plt.plot(x, CumSum, label="Watts Cumulative")
	# for prc,frac in zip(Percentages,labels[1:-1]):
	# 	plt.plot([frac, frac], [0, prc], '--',label="{}% of power".format(prc*100))
	# plt.legend(loc='lower right')
	# plt.xticks(labels)
	# plt.xlabel('Number of elements as fraction of total')
	# plt.ylabel('Scaled  power')
	# plt.show()

	return Watts, WattsPV, Elements

def ASCIIname(names):
	namelist = []
	for name in names:
		lis = [0]*80
		lis[:len(name)] = list(map(ord,name))
		namelist.append(lis)
	res = np.array(namelist)
	return res


def ERMES(Info, StudyDict):
	RunERMES = getattr(StudyDict['Parameters'], 'RunERMES', True)

	ERMESfile = '{}/ERMES.rmed'.format(StudyDict['PREASTER'])
	# Create a new set of ERMES results
	if RunERMES:
		Watts, WattsPV, Elements = SetupERMES(Info, StudyDict, ERMESfile)
	# Read in a previous set of ERMES results
	elif os.path.isfile(ERMESfile):
		pass
		NbVolumes = MeshInfo("{}/{}.med".format(Info.MESH_DIR,StudyDict['Parameters'].Mesh), meshname='Sample').NbVolumes
		ERMESres = h5py.File(ERMESfile, 'r')
		Watts = ERMESres["EM_Load/Watts"][:]
		WattsPV = ERMESres["EM_Load/WattsPV"][:]
		Elements = ERMESres["EM_Load/Elements"][:]
		# Check that the results match up with the mesh
		if Watts.shape[0] != NbVolumes:
			Info.Exit("EM.dat file doesn't match with current mesh")
	else :
		Info.Exit("ERMES results file '{}' does not exist and RunERMES flag not set to True".format(ERMESfile))

	CumSum = Watts.cumsum()
	CoilPower = CumSum[-1]
	print("Power delivered by coil: {:.4f}W".format(CoilPower*StudyDict['Parameters'].Current**2))

	# Find position in CumSum where the threshold percentage has been reached
	Threshold = StudyDict['Parameters'].EMThreshold
	pos = bl(CumSum,Threshold*CoilPower)

	print("To ensure {}% of the coil power is delivered {} elements will be assigned EM loads".format(Threshold*100, pos+1))

	EM_Val = WattsPV[:pos+1]*StudyDict['Parameters'].Current**2
	EM_Els = Elements[:pos+1]
	# Scale EM_Val to ensure correct energy input
	if getattr(StudyDict['Parameters'],'EMScale', False):
		EM_Val = EM_Val*(CumSum[-1]/CumSum[pos])

	EMLoadFile = '{}/ERMES.npy'.format(StudyDict['TMP_CALC_DIR'])
	np.save(EMLoadFile, np.vstack((EM_Els, EM_Val)).T)

	GroupBy = 'SALOME'


	tmpMeshFile = "{}/Mesh.med".format(StudyDict["TMP_CALC_DIR"])

	if GroupBy == 'H5PY':
		st = time.time()
		MeshMed = h5py.File(StudyDict['MeshFile'], 'r')
		tmpMeshMed = h5py.File(tmpMeshFile,'w')
		tmpMeshMed.copy(MeshMed["INFOS_GENERALES"],"INFOS_GENERALES")
		tmpMeshMed.copy(MeshMed["FAS/Sample"],"FAS/Sample")
		tmpMeshMed.copy(MeshMed["ENS_MAA/Sample"],"ENS_MAA/Sample")
		MeshMed.close()

		ElInfo = tmpMeshMed["ENS_MAA/Sample/-0000000000000000001-0000000000000000001/MAI/TE4"]
		ElList = ElInfo["NUM"][:]
		Elbool = np.searchsorted(ElList,EM_Els)
		ElList = ElList[Elbool]
		ElFam = ElInfo["FAM"][:][Elbool]
		UniqueFam = np.unique(ElFam)

		ElGrps = tmpMeshMed["FAS/Sample/ELEME"]
		MinNum, GrpName = 0, []
		for grp in ElGrps.keys():
			grpnum = ElGrps[grp].attrs['NUM']
			MinNum = min(MinNum,grpnum)
			if grpnum in UniqueFam:
				GrpName.append((grpnum,grp))

		Formats = h5py.File("{}/MED_Format.med".format(Info.COM_SCRIPTS),'r')
		NewNum = MinNum-1

		for Num, key in GrpName:
			st2 = time.time()
			NameGrps = ElGrps["{}/GRO/NOM".format(key)][:]
			NumGrps = NameGrps.shape[0]
			Fambool = ElFam == Num
			tmplist=[]
			dsetFormat = Formats["Name{}".format(NumGrps+2)]
			grpobj = ElGrps[key]
			for El in ElList[Fambool]:
				EMnames = ASCIIname(['EMgrp','M{}'.format(El)])
				NewNames = np.vstack((NameGrps,EMnames))
				newkey = "Grp{}".format(NewNum)

				ElGrps.copy(dsetFormat,"{}/GRO/NOM".format(newkey))
				ElGrps[newkey].attrs.create('NUM',NewNum,dtype='i4')
				ElGrps["{}/GRO".format(newkey)].attrs.create('NBR',NumGrps+2,dtype='i4')
				ElGrps["Grp{}/GRO/NOM".format(NewNum)][:] = NewNames

				# ElGrps.copy(grpobj,newkey)
				# ElGrps[newkey].attrs.modify('NUM',NewNum)
				# ElGrps["{}/GRO".format(newkey)].attrs.modify('NBR',NumGrps+2)
				# del ElGrps["{}/GRO/NOM".format(newkey)]
				# ElGrps.copy(dsetFormat,"{}/GRO/NOM".format(newkey))
				# ElGrps["{}/GRO/NOM".format(newkey)][:] = NewNames

				tmplist.append(NewNum)
				NewNum -=1

			ElFam[Fambool] = tmplist
			# print(time.time()-st2)

		Formats.close()
		ElFamFull = ElInfo["FAM"][:]
		ElFamFull[Elbool] = ElFam
		ElInfo["FAM"][:] = ElFamFull
		tmpMeshMed.close()
		StudyDict['MeshFile'] = tmpMeshFile
		print('Create:{}'.format(time.time()-st))

	elif GroupBy == 'SALOME':
		st = time.time()
		ArgDict = {"MeshFile":StudyDict["MeshFile"], "tmpMesh":tmpMeshFile,"EMLoadFile":EMLoadFile}
		EMGroupFile = "{}/CreateEMGroups.py".format(os.path.dirname(os.path.abspath(__file__)))
		Info.SalomeRun(EMGroupFile, ArgDict=ArgDict)
		StudyDict['MeshFile'] = tmpMeshFile
		print('Create:{}'.format(time.time()-st))



def main(Info, StudyDict):
	GetHTC(Info, StudyDict)
	if StudyDict['Parameters'].EMLoad == 'ERMES':
		ERMES(Info, StudyDict)
