import h5py
import numpy as np
import os
import sys
from VLFunctions import MeshInfo, MaterialProperty
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.ndimage

def KCalc(halfT, l, rho, Cp):
	alpha=(0.1388*(l**2))/halfT
	k=alpha*Cp*rho
	return k, alpha

def Individual(Info, StudyDict):
#	StudyDict = Info.Studies[study]

	ResFile = '{}/{}.rmed'.format(StudyDict['ASTER'], StudyDict['Parameters'].ResName)
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

	# Plot of Laser pulse used
	Laser = np.fromfile('{}/Laser/{}.dat'.format(Info.SIM_SCRIPTS,StudyDict['Parameters'].LaserT),dtype=float,count=-1,sep=" ")
	xLaser = Laser[::2]
	yLaser = Laser[1::2]
	fig = plt.figure(figsize = (10,5))
	plt.xlabel('Time',fontsize = 20)
	plt.ylabel('Laser Pulse\n Factor',fontsize = 20)
	plt.plot(xLaser, yLaser)
	plt.savefig("{}/LaserProfile.png".format(StudyDict['POSTASTER']), bbox_inches='tight')
	print("Created plot LaserProfile.png")
	plt.close()

	# Get plot of flux over top surface
	if StudyDict['Parameters'].LaserS in ('Gauss', 'gauss'):
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

	elif StudyDict['Parameters'].LaserS == 'Uniform':
		FluxRes = np.zeros((Top_Face.NbNodes,1))
		for face in Top_Face.Connect:
			FaceIx = np.where(np.in1d(Top_Face.Nodes, face))[0]
			Coords = TopCoor[FaceIx]
			area = 0.5*np.linalg.norm(np.cross(Coords[1] - Coords[0], Coords[2] - Coords[0]))
			FluxRes[FaceIx] += area/3

		FluxRes = np.hstack(( FluxRes, np.ones((Top_Face.NbNodes,1))))

	cmap=cm.coolwarm
	fig = plt.figure(figsize = (10,5))
	ax1 = plt.subplot(121,adjustable = 'box',aspect = 1)
	ax2 = plt.subplot(122,adjustable = 'box',aspect = 1)
	im1 = ax1.scatter(TopCoor[:,0], TopCoor[:,1], c=FluxRes[:,0], cmap=cmap)
	im2 = ax2.scatter(TopCoor[:,0], TopCoor[:,1], c=FluxRes[:,1], cmap=cmap)
	ax1.axis('off')
	ax2.axis('off')
	plt.savefig("{}/FluxDist.png".format(StudyDict['POSTASTER']), bbox_inches='tight')
	print("Created plot FluxDist.png")
	plt.close()

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

	ParaVis = {}
	BtmIx = Btm_Face.Nodes - 1
	Time, AvTemp, R = [], [], []
	for step in g[ResTher].keys():
		tm = g['{}/{}'.format(ResTher, step)].attrs['PDT']
		Time.append(tm)
		resTemp = g['{}/{}/NOE/MED_NO_PROFILE_INTERNAL/CO'.format(ResTher, step)][:]

		if tm == StudyDict['Parameters'].CaptureTime:
			ParaVis["Range"] = [min(resTemp), max(resTemp)]
			ParaVis["CaptureTime"] = tm

		# average temperatures for bottom surface (whole & R factor)
		BtmTemp = resTemp[BtmIx]
		AvTemp.append(np.average(BtmTemp))
		if Rplot:
			R.append(np.dot(Rbool, BtmTemp)/Rcount)

	# Plot temp v time plot for whole of bottom surface
	fig = plt.figure(figsize = (14,5))
	plt.xlabel('Time',fontsize = 20)
	plt.ylabel('Temperature',fontsize = 20)
	plt.plot(Time, AvTemp)
	plt.savefig("{}/BaseTemp.png".format(StudyDict['POSTASTER']), bbox_inches='tight')
	print("Created plot BaseTemp.png")
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
		plt.savefig("{}/Rplot.png".format(StudyDict['POSTASTER']), bbox_inches='tight')
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

		if VoidHeight == 0:
			CalcLambda, CalcAlpha = KCalc(HalfTime, Height, Rho, Cp)
			Lambdadat = np.fromfile('{}/Lambda.dat'.format(matpath),dtype=float,count=-1,sep=" ")
			Lambda = MaterialProperty(Lambdadat,StudyDict['Parameters'].InitTemp)
			print("Thermal conductivity of material ({}) used for simulation: {}W/mK".format(Materials, Lambda))
			print("Back calculated thermal conductivity: {} W/mK".format(CalcLambda))
			print("Error: {}%\n".format(100*abs(1-CalcLambda/Lambda)))
		else:
			print("Back calculation of the thermal conductivity is impossible due to void")

		# Check if temperature increase is correct
		if StudyDict['Parameters'].TopHTC == StudyDict['Parameters'].BottomHTC == 0:
			ActdT = AvTemp[-1] - AvTemp[0]
			vol = Radius**2*np.pi*Height - VoidRadius**2*np.pi*VoidHeight
			ExpdT = StudyDict['Parameters'].Energy/(vol*Rho*Cp)
			print("Anticipated temperature increase from energy input: {}{}C".format(ExpdT, u'\N{DEGREE SIGN}'))
			print("Measured temperature increase from simulation: {}{}C".format(ActdT, u'\N{DEGREE SIGN}'))
			print("Error: {}%\n".format(100*abs(1-ActdT/ExpdT)))
		else :
			print("Cannot measure temperature change due to BC")

	if StudyDict['Parameters'].LaserS in ('Gauss', 'gauss'):
		ExactMGD = 1 - np.exp(-0.5*(Radius/sigma)**2)
		AprxMGD =  sum(FluxRes[:,0])
		print("Exact volume under multivariate Gaussian distribution: {}".format(ExactMGD))
		print("The volume due to the spatial discretisation: {}".format(AprxMGD))
		print("Error: {}%\n".format(100*abs(1-ExactMGD/AprxMGD)))

	# Accuracy of temporal discretisation for laser pulse
	Lsrdat = np.fromfile('{}/Laser/{}.dat'.format(Info.SIM_SCRIPTS, StudyDict['Parameters'].LaserT),dtype=float,count=-1,sep=" ")
	xp, fp = Lsrdat[::2], Lsrdat[1::2]
	TimeSteps = np.fromfile('{}/TimeSteps.dat'.format(StudyDict['ASTER']),dtype=float,count=-1,sep=" ")
	TimeSteps = TimeSteps[TimeSteps <= xp[-1]]
	ExactLaser = np.trapz(fp, xp)
	AprxLaser = np.trapz(np.interp(TimeSteps,xp,fp), TimeSteps)
	print("Exact area under the laser pulse curve is: {}".format(ExactLaser))
	print("The area due to temporal discretisation (timestep size) is: {}".format(AprxLaser))
	print("Error: {}%\n".format(100*abs(1-ExactLaser/AprxLaser)))

	StudyDict['ParaVis'] = ParaVis
	print(StudyDict['ParaVis'])

def Combined(Info):
	GlobalRange = [np.inf, -np.inf]
	Simulations = []
	for Name, StudyDict in Info.Studies.items():
		Simulations.append(Name)
		StudyPV = StudyDict["ParaVis"]
		print(StudyPV)
		GlobalRange = [min(StudyPV["Range"][0],GlobalRange[0]), max(StudyPV["Range"][1],GlobalRange[1])]
	PVDict = {"Simulations": Simulations, "GlobalRange":GlobalRange}

	Info.WriteModule("{}/{}.py".format(Info.TMP_DIR, "PVParameters"), PVDict)
	Info.SalomeRun(os.path.abspath(__file__), SalomeGUI=False, AddPath=Info.TMP_DIR)

def ParaVis():
	import sys
	sys.dont_write_bytecode=True
	import salome
	import numpy as np
	from importlib import import_module
	import PVParameters

	import pvsimple
	pvsimple.ShowParaviewView()

	renderView1 = pvsimple.GetActiveViewOrCreate('RenderView')

	for Sim in PVParameters.Simulations:
		print(Sim)
		Parameters = import_module("{}.Parameters".format(Sim))
		PathVL = import_module("{}.PathVL".format(Sim))

		thermalrmed = pvsimple.MEDReader(FileName="{}/{}.rmed".format(PathVL.ASTER,Parameters.ResName))

		# show data in view
		thermalrmedDisplay = pvsimple.Show(thermalrmed, renderView1)
		thermalrmedDisplay.Representation = 'Surface'

		# current camera placement for renderView1
		camangle = 30
		camradius = 0.025
		renderView1.CameraPosition = [0, -camradius*np.cos(np.radians(camangle)), camradius*np.sin(np.radians(camangle))]
		renderView1.CameraFocalPoint = [0,0,-0.002]
		renderView1.CameraViewUp = [0,0,0]
		renderView1.CameraParallelScale = 0.000
		renderView1.Background = [1,1,1]  ### White Background

		animationScene1 = pvsimple.GetAnimationScene()
		animationScene1.AnimationTime = Parameters.CaptureTime
		# animationScene1.UpdateAnimationUsingDataTimeSteps()

		# set scalar coloring
		pvsimple.ColorBy(thermalrmedDisplay, ('POINTS', 'resther_TEMP', 'TEMP'))

		CBmin, CBmax = PVParameters.GlobalRange
		resther_TEMPLUT = pvsimple.GetColorTransferFunction('resther_TEMP')
		resther_TEMPLUT.NumberOfTableValues = 12
		resther_TEMPLUT.RescaleTransferFunction(CBmin, CBmax)
		resther_TEMPPWF = pvsimple.GetOpacityTransferFunction('resther_TEMP')
		resther_TEMPPWF.RescaleTransferFunction(CBmin, CBmax)

		ColorBar = pvsimple.GetScalarBar(resther_TEMPLUT, renderView1)
		BarLength = 0.7
		FontSize = 24
		ColorBar.WindowLocation = 'AnyLocation'
		#ColorBar.WindowLocation = 'LowerCenter'
		ColorBar.Orientation = 'Horizontal'
		ColorBar.ScalarBarLength = BarLength
		ColorBar.ScalarBarThickness = FontSize
		ColorBar.Position = [(1-BarLength)/2, 0.03]
		ColorBar.Title = 'Temperature'
		ColorBar.ComponentTitle = ''
		ColorBar.TitleFontSize = FontSize
		ColorBar.TitleColor = [0,0,0]
		ColorBar.TitleBold = 1
		ColorBar.LabelFontSize = FontSize
		ColorBar.LabelColor = [0,0,0]
		ColorBar.LabelBold = 1
		ColorBar.CustomLabels = list(np.round(np.linspace(CBmin, CBmax,7), 2))
		ColorBar.UseCustomLabels = 1
		ColorBar.AddRangeLabels = 1
		ColorBar.RangeLabelFormat = '%-#.2f'
		ColorBar.Visibility = 1

		pvsimple.SaveScreenshot("{}/Capture.png".format(PathVL.POSTASTER), renderView1, ImageResolution=[1400, 590], FontScaling='Do not scale fonts')
		pvsimple.Hide(thermalrmed, renderView1)
		print("Created image Capture.png")

		clip1 = pvsimple.Clip(Input=thermalrmed)
		clip1.ClipType.Normal = [0.0, -1.0, 0.0]
		clip1Display = pvsimple.Show(clip1, renderView1)
		ColorBar.Visibility = 1
		pvsimple.SaveScreenshot("{}/ClipCapture.png".format(PathVL.POSTASTER), renderView1, ImageResolution=[1400, 590], FontScaling='Do not scale fonts')
		pvsimple.Hide(clip1, renderView1)
		print("Created image ClipCapture.png")

		pvsimple.Show(thermalrmed, renderView1)
		pvsimple.ColorBy(thermalrmedDisplay, None)
		thermalrmedDisplay.Representation = 'Surface With Edges'
		thermalrmedDisplay.EdgeColor = [0.0, 0.0, 0.0]
		thermalrmedDisplay.DiffuseColor = [0.2, 0.75, 0.996078431372549]
		pvsimple.SaveScreenshot("{}/Mesh.png".format(PathVL.POSTASTER), renderView1, ImageResolution=[1400, 590], FontScaling='Do not scale fonts')
		pvsimple.Hide(thermalrmed, renderView1)
		print("Created image Mesh.png")

		calculator1 = pvsimple.Calculator(Input=thermalrmed)
		calculator1.ResultArrayName = 'Resulti'
		calculator1.Function = 'coords.iHat'
		calculator2 = pvsimple.Calculator(Input=calculator1)
		calculator2.ResultArrayName = 'Resultj'
		calculator2.Function = 'coords.jHat'
		calculator3 = pvsimple.Calculator(Input=calculator2)
		calculator3.ResultArrayName = 'Resultk'
		calculator3.Function = 'coords.kHat'
		threshold1 = pvsimple.Threshold(Input=calculator3)
		threshold1Display = pvsimple.Show(threshold1, renderView1)
		threshold1Display.Representation = 'Surface With Edges'
		threshold1.Scalars = ['POINTS', 'Resultj']
		threshold1.ThresholdRange = [0, 0.0063]
		threshold1Display.EdgeColor = [0.0, 0.0, 0.0]
		threshold1Display.DiffuseColor = [0.2, 0.75, 0.996078431372549]

		renderView1.CameraPosition = [0, -0.015, 0.00125]
		renderView1.CameraFocalPoint = [0, 0, 0.00125]
		renderView1.CameraViewUp = [0.0, 0.0, 1.0]

		pvsimple.SaveScreenshot("{}/MeshCrossSection.png".format(PathVL.POSTASTER), renderView1, ImageResolution=[1400, 590], FontScaling='Do not scale fonts')
		pvsimple.Hide(threshold1, renderView1)
		print("Created image MeshCrossSection.png")

		for source in pvsimple.GetSources().values():
				pvsimple.Delete(source)

		globals().update(locals())

if __name__=='__main__':
	ParaVis()