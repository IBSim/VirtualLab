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

def GetHTC(Info):
	for Name, StudyDict in Info.Studies.items():
		CreateHTC = getattr(StudyDict['Parameters'], 'CreateHTC', True)

		if CreateHTC == None: continue
		# Create a new set of HTC values
		if CreateHTC:
			from HTC.Coolant import Properties as ClProp
			from HTC.Pipe import PipeGeom
			from HTC.ITER import htc as htc_ITER

			Pipedict = StudyDict['Parameters'].Pipe
			Pipe = PipeGeom(shape=Pipedict['Type'], pipediameter=Pipedict['Diameter'], length=Pipedict['Length'])

			Cooldict = StudyDict['Parameters'].Coolant
			Coolant = ClProp(T=Cooldict['Temperature']+273, P=Cooldict['Pressure'], velocity=Cooldict['Velocity'])

			# Starting WallTemp and increment between temperatures to check
			WallTemp, incr = 5, 5
			HTC = []
			while True:
#				print(WallTemp)
				h = htc_ITER(Coolant, Pipe, WallTemp + 273)
				if h == 0: break
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

def SetupERMES(Info, StudyDict, **kwargs):
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

	ERMESdict = StudyDict['Parameters'].ERMES
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
	"ProblemType = 8pr;\n" + \
	"ProblemType = LE;\n" + \
	"ProblemFrequency = {};\n".format(ERMESdict['Frequency']*2*np.pi) + \
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
	"ProblemType = 8pr;\n" + \
	"ProblemType = LE;\n" + \
	"ProblemFrequency = {};\n".format(ERMESdict['Frequency']*2*np.pi) + \
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

	# Create variables for contact node information 9used in dat file 1 and 3
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

	SampleMesh = MeshInfo(MeshFile, meshname='Sample')

	JH_NL = [] # NonLinear JouleHeating
	for Temp in Temperatures:
		JHres, Jres = [], []
		JHstart, Jstart = 3, 3 + (NbNodesERMES + 3)
		JHend, Jend = JHstart + SampleMesh.NbNodes, Jstart + NbNodesERMES
		with open('{}/{}{}.post.res'.format(StudyDict['TMP_CALC_DIR'],'Wave',Temp),'r') as f:
			for j,line in enumerate(f):
				if JHstart <= j < JHend:
					JHres.append(float(line.split()[1]))
				elif Jstart <= j < Jend:
					Jres.append(float(line.split()[1]))
				elif j >= Jend:
					break

		facesum, intJ, intJsq = 0, 0, 0
		Jres = np.array(Jres)
		for nodes in CoilInCnct:
			coor1, coor2, coor3 = ERMESMesh.GetNodeXYZ(nodes)
			area = 0.5*np.linalg.norm(np.cross(coor2-coor1,coor3-coor1))
			facesum += area

			J1, J2, J3 = Jres[nodes - 1]
			intJ += area*(J1 + J2 + J3)/3
			intJsq += area*(J1**2 + J2**2 + J3**2)/3

		if check:
			print('These values should match up with those on the output from ERMES')
			print('Area: {:.6e}'.format(facesum))
			print('intSurf|J|: {:.6e}'.format(intJ))
			print('intSurf|J|^2: {:.6e}'.format(intJsq))

		ScaleFactor = (ERMESdict['Current']/intJ)




		JH_NL.append(np.array(JHres)*ScaleFactor**2)

	JH_Node = np.transpose(JH_NL)
#	
#	Nodes = list(range(1,SampleMesh.NbNodes+1))
#	Coor = SampleMesh.GetNodeXYZ(Nodes)
#	dataEM = np.hstack((Coor,JH_Node))
#	SumVol = 0
#	JH_Elem, Watts = [], []
#	for grp in StudyDict['Parameters'].Materials.keys():
#		grpinfo = SampleMesh.GroupInfo(grp)
#		for Nodes in grpinfo.Connect:
#			data = dataEM[Nodes-1,:]
#			Coors, EM = data[:,:3], data[:,3:]
#			ElSum = (np.sum(EM,axis=0)/4)
#			JH_Elem.append(ElSum)
#			vol = abs(np.dot(np.cross(Coors[1,:]-Coors[0,:],Coors[2,:]-Coors[0,:]),Coors[3,:]-Coors[0,:]))/float(6)
#			SumVol += vol
#			Watts.append(vol*ElSum)

#	JH_Elem.insert(0,np.array(Temperatures))
#	JH_Node = np.vstack((np.array(Temperatures), JH_Node))

#	return JH_Node, JH_Elem

	JH_Node = np.vstack((np.array(Temperatures), JH_Node))
	return JH_Node



def ERMES(Info):
	currdir = os.path.dirname(os.path.realpath(__file__))
	ERMESlist = []

	for Name, StudyDict in Info.Studies.items():
		RunERMES = getattr(StudyDict['Parameters'], 'RunERMES', True)

		if RunERMES == None: continue

		EMtype = 'Node'
		if EMtype == 'Elem': ERMESfname = 'ERMES_Elem.dat'
		else : ERMESfname = 'ERMES_Node.dat'
		EMpath = '{}/{}'.format(StudyDict['PREASTER'], ERMESfname)
		if RunERMES:
			### Create a new set of ERMES results
			JH_Node = SetupERMES(Info, StudyDict)
#			np.savetxt('{}/{}'.format(StudyDict['PREASTER_DIR'], 'ERMES_Elem.dat'), JH_Elem, fmt = '%.10f', delimiter = '   ')
			np.savetxt('{}/{}'.format(StudyDict['PREASTER'], 'ERMES_Node.dat'), JH_Node, fmt = '%.10f', delimiter = '   ')

		elif os.path.isfile(EMpath):
			SampleMesh = MeshInfo("{}/{}.med".format(Info.MESH_DIR,StudyDict['Parameters'].Mesh), meshname='Sample')
			EM = np.fromfile(EMpath, dtype=float, count=-1, sep=" ")
			if EMtype == 'Elem': NumCol = EM.shape[0]/(SampleMesh.NbVolumes+1)
			else : NumCol = EM.shape[0]/(SampleMesh.NbNodes+1)
			if  not NumCol.is_integer():
				Info.Exit("EM.dat file doesn't match with current mesh")
		else :
			Info.Exit('No EM.dat file found in OUTPUT_DIR and change RunEM not set to "yes"')

		shutil.copy(EMpath, StudyDict['TMP_CALC_DIR'])





	
#		NodeDat = EM[1:,:]*1e6
#		PerVol, Watts = [], []
#		EMMesh = VLFunctions.MeshInfo(EMmeshfile)
#		Nodes = list(range(1,EMMesh.NbNodes+1))
#		Coor = EMMesh.GetNodeXYZ(Nodes)
#		for grp in Info.Studies[study]['Parameters'].SampleGroups:
#			grpinfo = EMMesh.GroupInfo(grp)
#			for face in grpinfo.Connect:
#				Elsum = np.sum(NodeDat[face-1,:])/4
#				PerVol.append(Elsum)
#				VCoor = Coor[face-1]
#				vol = 1/float(6)*abs(np.dot(np.cross(VCoor[1,:]-VCoor[0,:],VCoor[2,:]-VCoor[0,:]),VCoor[3,:]-VCoor[0,:]))
#				Watts.append(vol*Elsum)


#		NbBins = 100
#		Whist, Wbins = np.histogram(Watts, bins=NbBins)
#		Total = sum(Whist)
#		print(Total)
#		print(Whist[0]/Total)

#		Whist, Wbins = np.histogram(PerVol, bins=NbBins)
#		Total = sum(Whist)
#		print(Total)
#		print(Whist[0]/Total)

#		for NbBins in range(5,115,10):
#			xPV, yPV, xW, yW, xNd, yNd = [], [], [], [], [], []
#			Ndhist, Ndbins = np.histogram(NodeDat, bins=NbBins)
#			PVhist, PVbins = np.histogram(PerVol, bins=NbBins)
#			Whist, Wbins = np.histogram(Watts, bins=NbBins)

#			Ndwidth = Ndbins[1] - Ndbins[0]
#			PVwidth = PVbins[1] - PVbins[0]
#			Wwidth = Wbins[1] - Wbins[0]
#			for i in range(NbBins):
#				xPV.append(0.5*(PVbins[i]+PVbins[i+1]))
#				yPV.append(PVhist[i]*PVbins[i])
#				xW.append(0.5*(Wbins[i]+Wbins[i+1]))
#				yW.append(Whist[i]*Wbins[i])
#				xNd.append(0.5*(Ndbins[i]+Ndbins[i+1]))
#				yNd.append(Ndhist[i]*Ndbins[i])

#			# Hist figures
#			fig = plt.figure(figsize = (14,5))
#			ax1 = plt.subplot(121,adjustable = 'box')
#			ax1.bar(xNd,Ndhist,Ndwidth)
#			ax1.set_title('W/m^3 nodal histogram')
#			ax2 = plt.subplot(122,adjustable = 'box')
#			ax2.bar(xPV,PVhist,PVwidth)
#			ax2.set_title('W/m^3 element histogram')
#			plt.tight_layout()
##			plt.show()
#			plt.savefig("{}/Hist/{}.png".format(StudyDict['OUTPUT_DIR'],NbBins), bbox_inches='tight')
#			plt.close()

#			# Hist2 figures
#			fig = plt.figure(figsize = (14,5))
#			ax1 = plt.subplot(321,adjustable = 'box')
#			ax1.bar(xNd,Ndhist,Ndwidth)
#			ax1.set_title('Node histogram')
#			ax2 = plt.subplot(322,adjustable = 'box')
#			ax2.bar(xNd,yNd,Ndwidth)
#			ax2.set_title('Node histogram scaled by bottom edge')
#			ax3 = plt.subplot(323,adjustable = 'box')
#			ax3.bar(xPV,PVhist,PVwidth)
#			ax3.set_title('W/m^3 histogram')
#			ax4 = plt.subplot(324,adjustable = 'box')
#			ax4.bar(xPV,yPV,PVwidth)
#			ax4.set_title('W/m^3 histogram scaled by bottom edge')
#			ax5 = plt.subplot(325,adjustable = 'box')
#			ax5.bar(xW,Whist,Wwidth)
#			ax5.set_title('W histogram')
#			ax6 = plt.subplot(326,adjustable = 'box')
#			ax6.bar(xW,yW,Wwidth)
#			ax6.set_title('W histogram scaled by bottom edge')
#			plt.tight_layout()
##			plt.show()
#			plt.savefig("{}/Hist2/{}.png".format(StudyDict['OUTPUT_DIR'],NbBins), bbox_inches='tight')
#			plt.close()

#		print('hello')
#		plt.show()

#		print(x, y)
#		plt.bar(x,y,width = bins[1]-bins[0])
#		plt.show()
#		plt.hist(Watts,bins = NbBins)
#		plt.show()



##		print('Nodal')
##		hist, bins = np.histogram(NodeDat, bins=NbBins)
##		print(hist)
##		print('Element')
##		hist, bins = np.histogram(ElemDat, bins=NbBins)
##		print(hist)
##		for NbBins in range(5,55,5):
#		for NbBins in [2]:
#			fig = plt.figure(figsize = (14,5))
#			ax1 = plt.subplot(121,adjustable = 'box')
#			ax2 = plt.subplot(122,adjustable = 'box')
#		
#			im1 = ax1.hist(NodeDat, bins=NbBins)
#			ax1.set_title('Nodal')
#			ax1.set_xlabel('JouleHeat')

#			im2 = ax2.hist(ElemDat, bins=NbBins)
#			ax2.set_title('Element')
#			ax2.set_xlabel('JouleHeat')

#			plt.savefig("{}/Hist/{}.png".format(StudyDict['OUTPUT_DIR'],NbBins), bbox_inches='tight')
#			plt.close()
#	




#	hist, bins = np.histogram(scaledres[1:])

			### Create the additional files necessary to run multiple ERMES simulation
#			ERMESFiles(Info,study)

			### Run the ERMES analysis
#			ErmesRun2(Info,study,NNodes)

#		### Read in previous EM results
#		elif os.path.isfile(Info.Studies[study]['DATA_DIR'] + '/EM.dat'):
#			EM = np.fromfile(Info.Studies[study]['DATA_DIR'] + '/EM.dat',dtype=float, count=-1, sep=" ")
#			EM = EM.reshape((NumNodes+1,len(EM)//(NumNodes+1)))
#			np.savetxt(Info.Studies[study]['TMP_CALC_DIR'] + '/EM.dat',EM,fmt = '%.8f')

#		### Exit due to no usable EM data
#		else :
#			Info.Exit('No EM.dat file found in DATA_DIR and change RunEM not set to "yes"')


def main(Info):
	GetHTC(Info)
	ERMES(Info)


	

