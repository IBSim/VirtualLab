import os
import sys
sys.dont_write_bytecode=True
import numpy as np
from salome.smesh import smeshBuilder
import salome_version
import SalomeFunc
import SMESH
import time

def SymmCoil(**kwargs):
	if salome_version.getVersions()[0] < 9:
		import salome
		theStudy = salome.myStudy
		smesh = smeshBuilder.New(theStudy)
	else :
		smesh = smeshBuilder.New()

	studydir = os.path.dirname(kwargs['paramfile'])
	sys.path.insert(0,studydir)
	Parameter = __import__(os.path.splitext(os.path.basename(kwargs['paramfile']))[0])

	([EMMesh], status) = smesh.CreateMeshesFromMED(kwargs['meshfile'])

	strNodes = "// List of nodes\n"
	for num in EMMesh.GetNodesId():
		Coord = EMMesh.GetNodeXYZ(num)
		strNodes = strNodes + "No[{}] = p({:.10f},{:.10f},{:.10f});\n".format(num,Coord[0],Coord[1],Coord[2])
	strNodes = strNodes + "\n"

	if kwargs['geomdir']:
		with open('{}/{}.dat'.format(kwargs['geomdir'],Parameter.MeshName),'w+') as f:		
			f.write(strNodes)

	del strNodes

	strMesh = "// Volume elements\n"
	strLoad = "// Source element\n"

	EMlist = ['Vacuum','Coil'] + Parameter.SampleGroups
	EMgrps = EMMesh.GetGroups()
	for i,name in enumerate(EMlist):
		### Find the group which corresponds to the name
		### Use GetGroupByName instead of this method
		for grp in EMgrps:
			if str(grp.GetName()) == name:
				NameMatch = True
				break

		if not NameMatch:
			print('The EM group name you specified is not defined in the mesh')

		else :
			elements = grp.GetListOfID()
			print ('Group {} has {} elements\n'.format(name,len(elements)))
			if name == 'Coil':
				for num in elements:
					ElNode = EMMesh.GetElemNodes(num)
					strMesh = strMesh + "VE({},{},{},{},{});\n".format(ElNode[0],ElNode[1],ElNode[2],ElNode[3],i+1)
					strLoad = strLoad + "JE({},{},{},{},{});\n".format(ElNode[0],ElNode[1],ElNode[2],ElNode[3],20)
			else: 
				for num in elements:
					ElNode = EMMesh.GetElemNodes(num)
					strMesh = strMesh + "VE({},{},{},{},{});\n".format(ElNode[0],ElNode[1],ElNode[2],ElNode[3],i+1)

	if kwargs['geomdir']:
		with open('{}/{}-1.dat'.format(kwargs['geomdir'],Parameter.MeshName),'w+') as f:
			f.write(strMesh)


		with open('{}/{}-2.dat'.format(kwargs['geomdir'],Parameter.MeshName),'w+') as f:
			f.write(strLoad)

def NonSymmCoil(ArgDict):
	if salome_version.getVersions()[0] < 9:
		import salome
		theStudy = salome.myStudy
		smesh = smeshBuilder.New(theStudy)
	else :
		smesh = smeshBuilder.New()

	studydir = ArgDict['StudyDir']
	Parameter = __import__(ArgDict['Parameters'])
	(Mesh, status) = smesh.CreateMeshesFromMED(ArgDict['MESH_FILE'])

	EMMesh = Mesh[1]

	NbNodes = EMMesh.NbNodes()
	NodeSt = NbNodes + 1

	CSnodes = (EMMesh.GetGroupByName('CoilSurface')[0]).GetNodeIDs()
	MSnodes = (EMMesh.GetGroupByName('SampleSurface')[0]).GetNodeIDs()
	DoubleNodes = MSnodes + CSnodes
	DoubleNodes = sorted(DoubleNodes)
	print(NbNodes)
	print(NbNodes+len(DoubleNodes))


	###### .dat file ######

	# Node part which will be in both Electrostatic and FullWave
	strNodes = "// List of nodes\n"
	for num in EMMesh.GetNodesId():
		Coord = EMMesh.GetNodeXYZ(num)
		strNodes += "No[{}] = p({:.10f},{:.10f},{:.10f});\n".format(num,Coord[0],Coord[1],Coord[2])

	for i,num in enumerate(DoubleNodes):
		Coord = EMMesh.GetNodeXYZ(num)
		strNodes += "No[{}] = p({:.10f},{:.10f},{:.10f});\n".format(NodeSt+i,Coord[0],Coord[1],Coord[2])


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
	"ProblemFrequency = {};\n".format(Parameter.Frequency*2*np.pi) + \
	"ProblemType = CheckConsistency;\n"	

	EMlist = ['Vacuum','Coil'] + Parameter.SampleGroups
	Electrolist = [0]*len(EMlist)
	Electrolist[1] = 1
	Stat02 = "// Material properties\n"
	for i,res in enumerate(Electrolist):
		Stat02 += "PROPERTIES[{}].IHL_ELECTRIC_CONDUCTIVITY  = {};\n".format(i+1,res) + \
		"PROPERTIES[{}].REAL_MAGNETIC_PERMEABILITY = {};\n".format(i+1,1) + \
		"PROPERTIES[{}].IMAG_MAGNETIC_PERMEABILITY = {};\n".format(i+1,0) + \
		"PROPERTIES[{}].REAL_ELECTRIC_PERMITTIVITY = {};\n".format(i+1,1) + \
		"PROPERTIES[{}].IMAG_ELECTRIC_PERMITTIVITY = {};\n".format(i+1,0)
#	Stat02 += '\n'

	Stat03 ="// Special materials properties\n" + \
	"PROPERTIES[17].COMPLEX_IBC = [0.000000000000000000,100.000000000000000000];\n" + \
	"PROPERTIES[32].COMPLEX_IBC = [1.0,0.0];\n"
#	Stat03 += '\n'

	Stat04 = "// Fixing static voltage on nodes in nodes\n"
	grpnodes = (EMMesh.GetGroupByName('CoilOut')[0]).GetNodeIDs()
	for num in grpnodes:
		Stat04 += "No[{}].V.Fix(0.0);\n".format(num)
#	Stat04 += '\n'

	Stat05 = "// Initializing building \n" + \
	"ElementsGroup = electromagnetic_group;\n\n" + \
	'// Generating debug results (if "Debug" mode activated) \n\n' + \
	"// Building and solving\n" + \
	"ProblemType = Build;\n"
#	Stat05 += '\n'

	with open('{}/Static.dat'.format(studydir),'w+') as f:
		f.write(Stat01 + strNodes + Stat02 + Stat03 + Stat04 + Stat05)


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
	"ProblemFrequency = {};\n".format(Parameter.Frequency*2*np.pi) + \
	"ProblemType = CheckConsistency;\n"

	Wave02 = "// Material properties\n"
#	for i,res in enumerate([0,0,100000]):
#		Wave02 += "PROPERTIES[{}].IHL_ELECTRIC_CONDUCTIVITY  = {};\n".format(i+1,res) + \
#		"PROPERTIES[{}].REAL_MAGNETIC_PERMEABILITY = {};\n".format(i+1,1) + \
#		"PROPERTIES[{}].IMAG_MAGNETIC_PERMEABILITY = {};\n".format(i+1,0) + \
#		"PROPERTIES[{}].REAL_ELECTRIC_PERMITTIVITY = {};\n".format(i+1,1) + \
#		"PROPERTIES[{}].IMAG_ELECTRIC_PERMITTIVITY = {};\n".format(i+1,0)


	for i, part in enumerate(EMlist):

		MgPrm = [1,0]
		ElPrm = [1,0]
		if part in ('Vacuum','Coil'):
			ElCnd = 0
		else :
			mat = getattr(Parameter,'{}Material'.format(part))
			
			fpath = '{}/{}/{}.dat'.format('/home/rhydian/Documents/Scripts/Simulation/virtuallab/Materials',mat,'ElecCond')
			prop = np.fromfile(fpath,dtype=float,count=-1,sep=" ")
			ElCnd = np.interp(20,prop[::2],prop[1::2])

		Wave02 += "PROPERTIES[{}].IHL_ELECTRIC_CONDUCTIVITY  = {};\n".format(i+1,ElCnd) + \
		"PROPERTIES[{}].REAL_MAGNETIC_PERMEABILITY = {};\n".format(i+1,MgPrm[0]) + \
		"PROPERTIES[{}].IMAG_MAGNETIC_PERMEABILITY = {};\n".format(i+1,MgPrm[1]) + \
		"PROPERTIES[{}].REAL_ELECTRIC_PERMITTIVITY = {};\n".format(i+1,ElPrm[0]) + \
		"PROPERTIES[{}].IMAG_ELECTRIC_PERMITTIVITY = {};\n".format(i+1,ElPrm[1])


	Wave03 = Stat03

	Wave04 = "// Creating High order nodes\n" + \
	"ProblemType = CreateHONodes;\n" + \
	"// Making contact elements\n" + \
	"ProblemType = MakeContact;\n" + \
	"// Fixing degrees of freedom in PEC nodes\n"

	PECEls = (EMMesh.GetGroupByName('VacuumSurface')[0]).GetListOfID()
	for face in PECEls:
		ElNode = EMMesh.GetElemNodes(face)
		Wave04 += "PEC = n([{},{},{}]);\n".format(ElNode[0],ElNode[1],ElNode[2])

	Wave05 = Stat05

	with open('{}/Wave.dat'.format(studydir),'w+') as f:
		f.write(Wave01 + strNodes + Wave02 + Wave03 + Wave04 + Wave05)

	print('dat file created')


	###### -1.dat file ######

	# Break up Vacuum group up in to those that border a surface and those that don't
	VacEls = (EMMesh.GetGroupByName('Vacuum')[0]).GetListOfID()
	els = []
	for node in DoubleNodes:
		els += EMMesh.GetElementsByNodes([node],SMESH.VOLUME)
	OnBound = set(els).intersection(VacEls)
	print(len(OnBound))
	OffBound = set(VacEls).difference(els)
	print(len(OffBound))

	#Loop through those which touch the surface, removing necessary nodes
	strMesh = "// Volume elements\n"
	
	SDoubleNodes = set(DoubleNodes)

	for num in OnBound:
		ElNode = EMMesh.GetElemNodes(num)
		common = SDoubleNodes.intersection(ElNode)

		for node in common:
			NewNode = NodeSt + DoubleNodes.index(node)
			ElNode[ElNode.index(node)] = NewNode

		strMesh = strMesh + "VE({},{},{},{},{});\n".format(ElNode[2],ElNode[1],ElNode[0],ElNode[3],1) 

	for num in OffBound:
		ElNode = EMMesh.GetElemNodes(num)
		strMesh = strMesh + "VE({},{},{},{},{});\n".format(ElNode[2],ElNode[1],ElNode[0],ElNode[3],1) 

	for i,name in enumerate(EMlist[1:]):
		elements = (EMMesh.GetGroupByName(name)[0]).GetListOfID()
		for num in elements:
			ElNode = EMMesh.GetElemNodes(num)
			# Elements must be in specific order due to different face orientation between Salome and GiD
			strMesh = strMesh + "VE({},{},{},{},{});\n".format(ElNode[2],ElNode[1],ElNode[0],ElNode[3],i+2) 
#	strMesh +='\n'

	with open('{}/Static-1.dat'.format(studydir),'w+') as f:
		f.write(strMesh)
	with open('{}/Wave-1.dat'.format(studydir),'w+') as f:
		f.write(strMesh)

	####### -2.dat file ######
	CoilList = []
	CoilInEls = (EMMesh.GetGroupByName('CoilIn')[0]).GetListOfID()

	Stat21 = "// Static Robin elements\n"
	for face in CoilInEls:
		ElNode = EMMesh.GetElemNodes(face)
		Stat21 += "GRC({},{},{},17);\n".format(ElNode[0],ElNode[1],ElNode[2])
		CoilList.append(ElNode)

	with open('{}/Static-2.dat'.format(studydir),'w+') as f:
		f.write(Stat21)

	with open('{}/Wave-2.dat'.format(studydir),'w+') as f:
		f.write("// Source elements\n")

	###### -3.dat file ######

	strCE = "// Contact elements\n"
	DoubleElems = (EMMesh.GetGroupByName('SampleSurface')[0]).GetListOfID() + (EMMesh.GetGroupByName('CoilSurface')[0]).GetListOfID()
	for face in DoubleElems:
		NewNodes = []
		ElNode = EMMesh.GetElemNodes(face)
		for nd in ElNode:
			New = NodeSt + DoubleNodes.index(nd)
			NewNodes.append(New)
		strCE += "CE = n([{},{},{},{},{},{}]);\n".format(ElNode[2],ElNode[1],ElNode[0],NewNodes[2],NewNodes[1],NewNodes[0])
		

	with open('{}/Wave-3.dat'.format(studydir),'w+') as f:
		f.write(strCE)


	###### -5.dat file ######

	CoilVols = (EMMesh.GetGroupByName('Coil')[0]).GetListOfID()

	volstr = "// Field integration over a surface of non-smoothed fields\n"
	facestr = "// Field integration over a surface\n"
	for vol in CoilVols:
		VolNode = EMMesh.GetElemNodes(vol)
		for FaceNode in CoilList:
			if all(x in VolNode for x in FaceNode):
				volstr += "PVIE({},{},{},{},0);\n".format(VolNode[0],VolNode[1],VolNode[2],VolNode[3])	
				facestr += "PSIE({},{},{},32);\n".format(FaceNode[0],FaceNode[1],FaceNode[2])

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


	with open('{}/Static-5.dat'.format(studydir),'w+') as f:
		f.write(Stat51 + volstr + facestr + Stat52)

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

	with open('{}/Wave-5.dat'.format(studydir),'w+') as f:
		f.write(Wave51 + facestr + Wave52)


	### -9.dat file
	name = '1'
	with open('{}/Static-9.dat'.format(studydir),'w+') as f:
		f.write('{}\n0\n'.format(name))
	with open('{}/Wave-9.dat'.format(studydir),'w+') as f:
		f.write('{}\n0\n'.format(name))

	print('Files Created')


if __name__ == '__main__':
	ArgDict = SalomeFunc.GetArgs(sys.argv[1:])
	print(ArgDict)
	if ArgDict['EMtype'] == 'NonSymmCoil':
		NonSymmCoil(ArgDict)

#	if sys.argv[1] == 'SymmCoil':
#		SymmCoil(paramfile=sys.argv[2],meshfile=sys.argv[3],geomdir=sys.argv[4])






