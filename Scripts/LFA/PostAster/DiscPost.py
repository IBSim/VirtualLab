import h5py
import numpy as np
import os
import sys
from VLFunctions import MeshInfo, MaterialProperty
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import scipy.ndimage
from importlib import import_module

def KCalc(halfT, l, rho, Cp):
	alpha=(0.1388*(l**2))/halfT
	k=alpha*Cp*rho
	return k, alpha

def Single(Info, StudyDict):
	Parameters = StudyDict["Parameters"]
	ResFile = '{}/Thermal.rmed'.format(StudyDict['ASTER'])

	# Get mesh information from the results file
	meshdata = MeshInfo(ResFile)
	Btm_Face = meshdata.GroupInfo('Bottom_Face')
	BtmCoor = meshdata.GetNodeXYZ(Btm_Face.Nodes)
	Top_Face = meshdata.GroupInfo('Top_Face')
	TopCoor = meshdata.GetNodeXYZ(Top_Face.Nodes)

	# If mesh info file exists import it
	if os.path.isfile("{}/{}.py".format(Info.MESH_DIR,Parameters.Mesh)):
		sys.path.insert(0,Info.MESH_DIR)
		VLMesh = import_module(Parameters.Mesh)
		Height = VLMesh.HeightT+VLMesh.HeightB
		TopCentre = np.array([0,0,Height])
		BtmCentre = np.array([0,0,0])
		Radius = VLMesh.Radius
		VoidHeight = abs(VLMesh.VoidHeight)
		VoidRadius = VLMesh.VoidRadius
	else :
		pass
		# Use the bottom and top surface to work out sample dimensions (don't want to rely on parameters file here)
		# BtmMin = BtmCoor.min(axis=0)
		# BtmMax = BtmCoor.max(axis=0)
		# BtmCentre = 0.5*(BtmMax + BtmMin)
		#
		# TopMin = TopCoor.min(axis=0)
		# TopMax = TopCoor.max(axis=0)
		# TopCentre = 0.5*(TopMax + TopMin)
		#
		# HeightVec = TopCentre - BtmCentre
		# HeightAx = np.argmax(HeightVec) # 0 is x, 1 is y, 2 is z
		# RadAx = [1,2,3]
		# RadAx.pop(HeightAx)
		# Height = np.linalg.norm(HeightVec)
		# Radius = 0.5*(BtmMax - BtmMin)[RadAx[0]]
		# if 'Void_Ext' in meshdata.GroupNames():
		# 	VoidExt = meshdata.GroupInfo('Void_Ext')
		# 	VoidCoor = meshdata.GetNodeXYZ(VoidExt.Nodes)
		# 	Diff = VoidCoor.max(axis=0) - VoidCoor.min(axis=0)
		# 	VoidHeight = Diff[HeightAx]
		# 	VoidRadius = Diff[RadAx[0]]/2
		# else :
		# 	VoidHeight = 0
		# 	VoidRadius = 0

	# Get plot of flux over top surface
	if Parameters.LaserS.lower() in ('g','gauss'):
		sigma = 0.005/(-2*np.log(1-0.1))**0.5
		FluxRes = np.zeros((Top_Face.NbNodes,2))
		for face in Top_Face.Connect:
			FaceIx = np.where(np.in1d(Top_Face.Nodes, face))[0]
			Coords = TopCoor[FaceIx]
			area = 0.5*np.linalg.norm(np.cross(Coords[1] - Coords[0], Coords[2] - Coords[0]))
			for Ix, ndcoor in zip(FaceIx, Coords):
				dist = np.linalg.norm(ndcoor - TopCentre)
				Gauss = (1/(2*np.pi*sigma**2))*np.exp(-dist**2/(2*sigma**2))
				FluxRes[Ix,:] += np.array([Gauss*area/3, area/3])
		FluxRes[:,1] = FluxRes[:,0]/FluxRes[:,1]
	else:
		FluxRes = np.zeros((Top_Face.NbNodes,1))
		for face in Top_Face.Connect:
			FaceIx = np.where(np.in1d(Top_Face.Nodes, face))[0]
			Coords = TopCoor[FaceIx]
			area = 0.5*np.linalg.norm(np.cross(Coords[1] - Coords[0], Coords[2] - Coords[0]))
			FluxRes[FaceIx] += area/3
		FluxRes = np.hstack(( FluxRes, np.ones((Top_Face.NbNodes,1))))

	Laser = np.fromfile('{}/Laser/{}.dat'.format(Info.SIM_SCRIPTS,Parameters.LaserT),dtype=float,count=-1,sep=" ")
	xLaser = Laser[::2]
	yLaser = Laser[1::2]

	gs = gridspec.GridSpec(2, 2,hspace=None,height_ratios=[1, 2])
	cmap=cm.coolwarm

	fig = plt.figure(figsize=(10,10))
	ax1 = fig.add_subplot(gs[0, :])
	ax1.plot(xLaser, yLaser)
	ax1.set_title('Temporal profile', fontsize = 14)
	ax1.set_xlabel('Time',fontsize = 12)
	ax1.set_ylabel('Normalised Pulse',fontsize = 12)
	ax1.set_ylim([0,1.1])
	ax1.set_xlim([xLaser[0],xLaser[-1]])

	ax2 = fig.add_subplot(gs[1, 0],adjustable = 'box',aspect = 1)
	ax2.scatter(TopCoor[:,0], TopCoor[:,1], c=FluxRes[:,1], cmap=cmap)
	ax2.set_title('Flux',fontsize = 14)
	ax2.axis('off')

	ax3 = fig.add_subplot(gs[1, 1],adjustable = 'box',aspect = 1)
	ax3.scatter(TopCoor[:,0], TopCoor[:,1], c=FluxRes[:,0], cmap=cmap)
	ax3.set_title('Nodal load',fontsize = 14)
	ax3.axis('off')

	plt.savefig("{}/LaserProfilel.png".format(StudyDict['POSTASTER']), bbox_inches='tight')
	print("Created plot LaserProfile.png")
	plt.close()

	# Find which nodes sit within the specified R values
	Rvalues = getattr(Parameters, 'Rvalues', [])
	if Rvalues:
		Rvalues = sorted(Rvalues, reverse = True)
		# remove R=1 as this will be included automatically in the Rfactor plot
		if Rvalues[0] == 1: Rvalues.pop(0)
		# Scale nodal values to a range of 0 to 1
		ScaleNorm = np.linalg.norm(BtmCoor - BtmCentre, axis=1)/Radius
		Rbool = [ScaleNorm <= R for R in Rvalues]
		Rbool = np.array(Rbool)
		Rcount = np.sum(Rbool,axis=1)

	# Open res file using h5py
	g = h5py.File(ResFile, 'r')
	gRes = g['/CHA/Temperature']

	ParaVis = {}
	BtmIx = Btm_Face.Nodes - 1
	AvTemp, R = [], []
	Steps = gRes.keys()
	Time = [gRes[step].attrs['PDT'] for step in Steps]
	CaptureTime = min(Time, key=lambda x:abs(x-Parameters.CaptureTime))

	ParaVis["CaptureTime"]=CaptureTime
	for tm,step in zip(Time,Steps):
		resTemp = gRes['{}/NOE/MED_NO_PROFILE_INTERNAL/CO'.format(step)][:]
		if tm==CaptureTime:
			ParaVis["Range"] = [min(resTemp), max(resTemp)]

		# average temperatures for bottom surface (whole & R factor)
		BtmTemp = resTemp[BtmIx]
		AvTemp.append(np.average(BtmTemp))
		if Rvalues:
			R.append(np.dot(Rbool, BtmTemp)/Rcount)

	StudyDict['ParaVis'] = ParaVis

	# Plot of average temperature on bottom surface
	fig = plt.figure(figsize = (14,5))
	plt.xlabel('Time',fontsize = 20)
	plt.ylabel('Temperature',fontsize = 20)
	plt.plot(Time, AvTemp, label = 'Avg. Temperature')
	for Rdata, Rval in zip(np.transpose(R), Rvalues):
		plt.plot(Time, Rdata, label = 'Avg. Temperature (R={})'.format(Rval))
	plt.legend(loc='upper left')
	plt.savefig("{}/AvgTempBase.png".format(StudyDict['POSTASTER']), bbox_inches='tight')
	print("Created plot AvgTempBase.png\n")
	plt.close()


	LaserStr="### Laser pulse discretisation###\n\n"
	# Accuracy of temporal discretisation for laser pulse
	TimeSteps = np.fromfile('{}/TimeSteps.dat'.format(StudyDict['ASTER']),dtype=float,count=-1,sep=" ")
	TimeSteps = TimeSteps[TimeSteps <= xLaser[-1]]
	ExactLaser = np.trapz(yLaser, xLaser)
	AprxLaser = np.trapz(np.interp(TimeSteps,xLaser,yLaser), TimeSteps)
	Error = 100*abs(1-ExactLaser/AprxLaser)
	LaserStr += "Exact area under laser temporal profile: {:.6f}\n"\
				"Area due to temporal discretisation: {:.6f}\n"\
				"Error: {:.6f}%\n\n".format(ExactLaser,AprxLaser,Error)

	if Parameters.LaserS.lower() in ('g','gauss'):
		ExactMGD = 1 - np.exp(-0.5*(Radius/sigma)**2)
		AprxMGD =  sum(FluxRes[:,0])
		Error = 100*abs(1-ExactMGD/AprxMGD)
		LaserStr += "Exact volume under laser spatial profile (MGD): {:.6f}\n"\
					"Volume due to the spatial discretisation: {:.6f}\n"\
					"Error: {:.6f}%\n".format(ExactMGD,AprxMGD,Error)
	else:
		LaserStr += "Spatial profile exact due to uniform profile\n"

	LaserStr += "\n"

	TCStr = "### Thermal Conductivity ###\n\n"
	TempStr = "### Anticipated temperature ###\n\n"

	# Half Rise time of sample
	HalfTime = np.interp(0.5*(AvTemp[0]+AvTemp[-1]), AvTemp, Time)

	# Check thermal conductivity if there's no void and it's all the same material
	Materials = list(set(Parameters.Materials.values()))
	if len(Materials)==1:
		matpath = "{}/{}".format(Info.MATERIAL_DIR, Materials[0])
		Rhodat = np.fromfile('{}/Rho.dat'.format(matpath),dtype=float,count=-1,sep=" ")
		Rho = MaterialProperty(Rhodat,Parameters.InitTemp)
		Cpdat = np.fromfile('{}/Cp.dat'.format(matpath),dtype=float,count=-1,sep=" ")
		Cp = MaterialProperty(Cpdat,Parameters.InitTemp)

		if VoidHeight==0:
			CalcLambda, CalcAlpha = KCalc(HalfTime, Height, Rho, Cp)
			Lambdadat = np.fromfile('{}/Lambda.dat'.format(matpath),dtype=float,count=-1,sep=" ")
			Lambda = MaterialProperty(Lambdadat,Parameters.InitTemp)
			Error = 100*abs(1-CalcLambda/Lambda)
			TCStr += "Thermal conductivity of {}: {:.3f}W/mK\n"\
					 "Calculated thermal conductivity from results: {:.3f} W/mK\n"\
					 "Error: {:.6f}%\n".format(Materials[0], Lambda, CalcLambda,Error)
		else:
			TCStr += "Thermal conductivity calculation impossible due to void\n"

		if Parameters.TopHTC==Parameters.BottomHTC==0:
			ActdT = AvTemp[-1] - AvTemp[0]
			vol = Radius**2*np.pi*Height - VoidRadius**2*np.pi*VoidHeight
			ExpdT = Parameters.Energy/(vol*Rho*Cp)
			Error = 100*abs(1-ActdT/ExpdT)
			TempStr += "Measured temperature increase from simulation: {0:.6f}{1}C\n"\
					   "Anticipated temperature increase from energy input: {2:.6f}{1}C\n"\
					   "Error: {3:.6f}%\n".format(ActdT,u'\N{DEGREE SIGN}',ExpdT,Error)
		else :
			TempStr+= "Anticipated temperature change calculation impossible due to BC\n"
	else:
		TCStr += "Thermal conductivity calculation impossible due to multiple materials\n"
		TempStr += "Anticipated temperature change calculation impossible due to multiple materials\n"

	TCStr +="\n"
	TempStr+="\n"

	with open("{}/Summary.txt".format(StudyDict["POSTASTER"]),'w') as f:
		f.write(LaserStr+TCStr+TempStr)

def Combined(Info):
	GlobalRange = [np.inf, -np.inf]
	ArgDict = {}
	for Name, StudyDict in Info.SimData.items():
		StudyPV = StudyDict["ParaVis"]
		ArgDict[Name] = StudyPV["CaptureTime"]
		GlobalRange = [min(StudyPV["Range"][0],GlobalRange[0]), max(StudyPV["Range"][1],GlobalRange[1])]
	ArgDict['Rangemin'] = GlobalRange[0]
	ArgDict['Rangemax'] = GlobalRange[1]

	print('Creating images using ParaViS')
	GUI = getattr(Info.Parameters_Master.Sim, 'PVGUI', False)
	ParaVisFile = "{}/ParaVis.py".format(os.path.dirname(os.path.abspath(__file__)))
	RC = Info.Salome.Run(ParaVisFile, GUI=GUI, AddPath=Info.TEMP_DIR, Args=ArgDict)
	if RC:
		return "Error in Salome run"
