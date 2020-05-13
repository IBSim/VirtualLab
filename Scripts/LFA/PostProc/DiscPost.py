import h5py
import numpy as np
import os
import sys
from VLFunctions import MeshInfo, MaterialProperty
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from fpdf import FPDF
import scipy.ndimage


def KCalc(halfT, l, rho, Cp):
	alpha=(0.1388*(l**2))/halfT
	k=alpha*Cp*rho
	return k, alpha

def main(Info, StudyDict):
#	StudyDict = Info.Studies[study]

	ResFile = '{}/{}.rmed'.format(StudyDict['ASTER_DIR'], StudyDict['Parameters'].ResName)
	# Get mesh information from the results file
	meshdata = MeshInfo(ResFile)

	# Use the bottom and top surface to work out sample dimensions (don't want to rely on parameters file here)
	Btm_Face = meshdata.GroupInfo('Bottom_Face')
	BtmCoor = meshdata.GetNodeXYZ(Btm_Face.Nodes)
	BtmMin = BtmCoor.min(axis=0)
	BtmMax = BtmCoor.max(axis=0)
	BtmCentre = 0.5*(BtmMax + BtmMin)

	Top_Face = meshdata.GroupInfo('Top_Face')
	TopCoor = meshdata.GetNodeXYZ(Top_Face.Nodes)
	TopMin = TopCoor.min(axis=0)
	TopMax = TopCoor.max(axis=0)
	TopCentre = 0.5*(TopMax + TopMin)

	HeightVec = TopCentre - BtmCentre
	HeightAx = np.argmax(HeightVec) # 0 is x, 1 is y, 2 is z
	RadAx = [1,2,3]
	RadAx.pop(HeightAx)
	Height = np.linalg.norm(HeightVec)

	Radius = 0.5*(BtmMax - BtmMin)[RadAx[0]]
	
	if 'Void_Ext' in meshdata.GroupNames():
		VoidExt = meshdata.GroupInfo('Void_Ext')
		VoidCoor = meshdata.GetNodeXYZ(VoidExt.Nodes)
		Diff = VoidCoor.max(axis=0) - VoidCoor.min(axis=0)
		VoidHeight = Diff[HeightAx]
		VoidRadius = Diff[RadAx[0]]/2
	else :
		VoidHeight = 0
		VoidRadius = 0 

#	print(Radius, Height, VoidRadius, VoidHeight)

	# Create plots of temperature v time on the bottom surface
	Rvalues = getattr(StudyDict['Parameters'], 'Rvalues', None)
	if Rvalues:
		Rplot = True
		Rvalues = sorted(Rvalues, reverse = True)
		# remove R=1 as this will be included automatically in the Rfactor plot
		if Rvalues[0] == 1: Rvalues.pop(0)

		# Scale nodal values to a range of 0 to 1 
		ScaleNorm = np.linalg.norm(BtmCoor - BtmCentre, axis=1)/Radius
		Rbool = [ScaleNorm <= R for R in Rvalues]
		Rbool = np.array(Rbool)
		Rcount = np.sum(Rbool,axis=1)
	else:
		Rplot = False

	# Open res file using h5py
	g = h5py.File(ResFile, 'r')
	ResTher = '/CHA/resther_TEMP'

	BtmIx = Btm_Face.Nodes - 1
	Time, AvTemp, R = [], [], []
	for i, step in enumerate(g[ResTher].keys()):
		Time.append(g['{}/{}'.format(ResTher, step)].attrs['PDT'])
		grpvals = g['{}/{}/NOE/MED_NO_PROFILE_INTERNAL/CO'.format(ResTher, step)][:][BtmIx]
		# get average value over whole bottom surface
		AvTemp.append(np.average(grpvals))
		# get average over R factor of bottom surface
		if Rplot:
			Ravg = np.dot(Rbool, grpvals)/Rcount
			R.append(Ravg)

	# Plot temp v time plot for whole of bottom surface
	fig = plt.figure(figsize = (14,5))
	plt.xlabel('Time',fontsize = 20)
	plt.ylabel('Temperature',fontsize = 20)
	plt.plot(Time, AvTemp)
	plt.savefig("{}/BaseTemp.png".format(StudyDict['POST_DIR']), bbox_inches='tight')
	print("Created plot BaseTemp.png\n")
	plt.close()

	# Plot temp v time for rfactor bottom surface
	if Rplot:
		# Plot for different Rvalues, if desired
		fig = plt.figure(figsize = (14,5))
		plt.xlabel('Time',fontsize = 20)
		plt.ylabel('Temperature',fontsize = 20)	
		plt.plot(Time, AvTemp, label = 'Total area (R = 1)')
		for Rdata, Rval in zip(np.transpose(R), Rvalues):
			plt.plot(Time, Rdata, label = 'R = {}'.format(Rval))
		plt.legend(loc='upper left')
		plt.savefig("{}/Rplot.png".format(StudyDict['POST_DIR']), bbox_inches='tight')
		print("Created plot Rplot.png\n")
		plt.close()

	# Half Rise time of sample
	HalfTime = np.interp(0.5*(AvTemp[0] + AvTemp[-1]), AvTemp, Time)

	# Check thermal conductivity if there's no void and it's all the same material
	Materials = list(set(StudyDict['Parameters'].Materials.values()))	
	if len(Materials) == 1:
		matpath = "{}/{}".format(Info.MATERIAL_DIR, Materials[0])
		Rhodat = np.fromfile('{}/Rho.dat'.format(matpath),dtype=float,count=-1,sep=" ")
		Cpdat = np.fromfile('{}/Cp.dat'.format(matpath),dtype=float,count=-1,sep=" ")
		Rho = MaterialProperty(Rhodat,StudyDict['Parameters'].InitTemp)
		Cp = MaterialProperty(Cpdat,StudyDict['Parameters'].InitTemp)
		# Check thermal conductivity if there's no void
		print("### Thermal conductivity validation ###")
		if VoidHeight == 0:
			CalcLambda, CalcAlpha = KCalc(HalfTime, Height, Rho, Cp)	
			Lambdadat = np.fromfile('{}/Lambda.dat'.format(matpath),dtype=float,count=-1,sep=" ")
			Lambda = MaterialProperty(Lambdadat,StudyDict['Parameters'].InitTemp)
			print("Thermal conductivity of material ({}) used for simulation: {}W/mK".format(Materials, Lambda))
			print("Back calculated thermal conductivity: {} W/mK".format(CalcLambda))
			print("Error: {}%".format(100*abs(1-CalcLambda/Lambda)))
		else:
			print("Back calculation of the thermal conductivity is impossible due to void")
		print('#######################################\n')

		print("### Temperature increase validation ###")
		# Check if temperature increase is correct
		if StudyDict['Parameters'].TopHTC == StudyDict['Parameters'].BottomHTC == 0:
			ActdT = AvTemp[-1] - AvTemp[0]
			vol = Radius**2*np.pi*Height - VoidRadius**2*np.pi*VoidHeight
			ExpdT = StudyDict['Parameters'].Energy/(vol*Rho*Cp)
			print("Anticipated temperature increase from energy input: {}{}C".format(ActdT, u'\N{DEGREE SIGN}'))
			print("Measured temperature increase from simulation: {}{}C\n".format(ExpdT, u'\N{DEGREE SIGN}'))
			print("Error: {}%".format(100*abs(1-ActdT/ExpdT)))
		else :
			print("Cannot measure temperature change due to BC")
		print('#######################################\n')

	# Get plot of flux over top surface
	if StudyDict['Parameters'].LaserS in ('Gauss', 'gauss'):
		sigma = 0.005/(-2*np.log(1-0.1))**0.5
		Res = [np.array([0.0, 0.0]) for i in range(Top_Face.NbNodes)]
		for face in Top_Face.Connect:
			FaceIx = np.where(np.in1d(Top_Face.Nodes, face))[0]
			Coords = TopCoor[FaceIx]
			area = 0.5*np.linalg.norm(np.cross(Coords[1] - Coords[0], Coords[2] - Coords[0]))

			for Ix, ndcoor in zip(FaceIx, Coords):	
				dist = np.linalg.norm(ndcoor - TopCentre)
				Gauss = (1/(2*np.pi*sigma**2))*np.exp(-dist**2/(2*sigma**2))
				Res[Ix] += np.array([Gauss*area/3, area/3])

		Res = np.array(Res)
		Res[:,1] = Res[:,0]/Res[:,1]
		ExactMGD = 1 - np.exp(-0.5*(Radius/sigma)**2)
		AprxMGD =  sum(Res[:,0])
		print("### Accuracy of Gaussian profile ###")
		print("Exact volume under multivariate Gaussian distribution: {}".format(ExactMGD))
		print("The volume due to the spatial discretisation: {}\n".format(AprxMGD))
		print("Error: {}%".format(100*abs(1-ExactMGD/AprxMGD)))
		print("####################################\n")

	elif StudyDict['Parameters'].LaserS == 'Uniform':
		Res = np.zeros((Top_Face.NbNodes,1))
		for face in Top_Face.Connect:
			FaceIx = np.where(np.in1d(Top_Face.Nodes, face))[0]
			Coords = TopCoor[FaceIx]
			area = 0.5*np.linalg.norm(np.cross(Coords[1] - Coords[0], Coords[2] - Coords[0]))
			Res[FaceIx] += area/3

		Res = np.hstack(( Res, np.ones((Top_Face.NbNodes,1))))

	cmap=cm.coolwarm
	fig = plt.figure(figsize = (10,5))
	ax1 = plt.subplot(121,adjustable = 'box',aspect = 1)
	ax2 = plt.subplot(122,adjustable = 'box',aspect = 1)
	im1 = ax1.scatter(TopCoor[:,0], TopCoor[:,1], c=Res[:,0], cmap=cmap)
	im2 = ax2.scatter(TopCoor[:,0], TopCoor[:,1], c=Res[:,1], cmap=cmap)
	ax1.axis('off')
	ax2.axis('off')
	plt.savefig("{}/FluxDist.png".format(StudyDict['POST_DIR']), bbox_inches='tight')
	print("Created plot FluxDist.png")
	plt.close()

	# Accuracy of temporal discretisation for laser pulse
	Lsrdat = np.fromfile('{}/Laser/{}.dat'.format(Info.SIM_SCRIPTS, StudyDict['Parameters'].LaserT),dtype=float,count=-1,sep=" ")
	xp, fp = Lsrdat[::2], Lsrdat[1::2]
	TimeSteps = np.fromfile('{}/TimeSteps.dat'.format(StudyDict['ASTER_DIR']),dtype=float,count=-1,sep=" ")
	TimeSteps = TimeSteps[TimeSteps <= xp[-1]]
	ExactLaser = np.trapz(fp, xp)
	AprxLaser = np.trapz(np.interp(TimeSteps,xp,fp), TimeSteps) 
	print("### Accuracy of Laser profile ###")
	print("Exact area under the laser pulse curve is: {}".format(ExactLaser))
	print("The area due to temporal discretisation (timestep size) is: {}".format(AprxLaser))
	print("Error: {}%".format(100*abs(1-ExactLaser/AprxLaser)))
	print("#################################\n")

	# Plot of Laser pulse used
	Laser = np.fromfile('{}/Laser/{}.dat'.format(Info.SIM_SCRIPTS,StudyDict['Parameters'].LaserT),dtype=float,count=-1,sep=" ")
	xLaser = Laser[::2]
	yLaser = Laser[1::2]
	fig = plt.figure(figsize = (10,5))
	plt.xlabel('Time',fontsize = 20)
	plt.ylabel('Laser Pulse\n Factor',fontsize = 20)	
	plt.plot(xLaser, yLaser)
	plt.savefig("{}/LaserProfile.png".format(StudyDict['POST_DIR']), bbox_inches='tight')
	print("Created plot LaserProfile.png")
	plt.close()








