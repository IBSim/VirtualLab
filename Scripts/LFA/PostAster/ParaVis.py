
import sys
sys.dont_write_bytecode=True
import salome
import numpy as np
from importlib import import_module
import PVParameters
'''
import SalomePyQt
SalomePyQt.SalomePyQt().activateModule('ParaViS')
'''
import pvsimple
pvsimple.ShowParaviewView()

renderView1 = pvsimple.GetActiveViewOrCreate('RenderView')
for Sim in PVParameters.Simulations:
	print(Sim)
	Parameters = import_module("{}.Parameters".format(Sim))
	PathVL = import_module("{}.PathVL".format(Sim))

	thermalrmed = pvsimple.MEDReader(FileName="{}/Thermal.rmed".format(PathVL.ASTER))
	pvsimple.RenameSource(Sim,thermalrmed)

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
	if Parameters.CaptureTime > animationScene1.EndTime:
		animationScene1.AnimationTime = animationScene1.EndTime
	else :
		animationScene1.AnimationTime = Parameters.CaptureTime
	# animationScene1.UpdateAnimationUsingDataTimeSteps()

	# set scalar coloring
	pvsimple.ColorBy(thermalrmedDisplay, ('POINTS', 'Temperature'))

	CBmin, CBmax = PVParameters.GlobalRange
	resther_TEMPLUT = pvsimple.GetColorTransferFunction('Temperature')
	resther_TEMPLUT.NumberOfTableValues = 12
	resther_TEMPLUT.RescaleTransferFunction(CBmin, CBmax)
	resther_TEMPPWF = pvsimple.GetOpacityTransferFunction('Temperature')
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

#	for source in pvsimple.GetSources().values():
#			pvsimple.Delete(source)
