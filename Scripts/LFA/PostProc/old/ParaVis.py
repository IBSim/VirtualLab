TMP_FILE = 'TMP_FILE_REPLACE'
#TMP_DIR = path.dirname(TMP_FILE)
#TMP_FILE = '/tmp/RhydianAster/Case2/Calc_4/tmpfile.txt'

f = open(TMP_FILE,'a+')
Files = f.readlines()
f.close()
Dic = {}
for line in Files:
	data = line.split()
	Dic[data[0][:-1]] = data[1]
OUTPUT_DIR = Dic['OUTPUT_DIR']
SCRIPT_DIR = Dic['SCRIPT_DIR']
IMAGE_DIR = Dic['IMAGE_DIR']
RESULTS_DIR = Dic['RESULTS_DIR']
RESULTS_FILE = Dic['RESULTS_FILE']

g = open(OUTPUT_DIR+'/Information.dat','a+')
Info = g.readlines()
g.close()
Dic = {}
for line in Info:
	data = line.split()
	Dic[data[0][:-1]] = data[1]
Nearest_Tstep = Dic['Nearest_Tstep']

####################################################

import sys
import salome
import numpy as np

salome.salome_init()

theStudy = salome.myStudy

import salome_notebook
notebook = salome_notebook.NoteBook(theStudy)


###
### PARAVIS component
###

import pvsimple
pvsimple.ShowParaviewView()

#### import the simple module from the paraview
from pvsimple import *
#### disable automatic camera reset on 'Show'
pvsimple._DisableFirstRenderCameraReset()

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [1403, 591]


thermalrmed = MEDReader(FileName=RESULTS_DIR + RESULTS_FILE)

# show data in view
thermalrmedDisplay = Show(thermalrmed, renderView1)

# trace defaults for the display properties.
thermalrmedDisplay.Representation = 'Surface'

# set scalar coloring
ColorBy(thermalrmedDisplay, ('POINTS', 'resther_TEMP', 'TEMP'))

# rescale color and/or opacity maps used to include current data range
thermalrmedDisplay.RescaleTransferFunctionToDataRange(False)

CBMin = 20
CBMax = 35
# get color transfer function/color map for 'resther_TEMP' (important one for scaling)
resther_TEMPLUT = GetColorTransferFunction('resther_TEMP')
resther_TEMPLUT.RescaleTransferFunction(CBMin, CBMax)
resther_TEMPLUT.NumberOfTableValues = 12
ImportPresets(filename=SCRIPT_DIR + '/PostProc/ColorBar.json')
resther_TEMPLUT.ApplyPreset('Black-Body Radiation Rhydian', True)

ColorBar = GetScalarBar(resther_TEMPLUT, renderView1)
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
ColorBar.CustomLabels = list(np.linspace(CBMin,CBMax,6))
ColorBar.UseCustomLabels = 1
ColorBar.AddRangeLabels = 0

# get opacity transfer function/opacity map for 'resther_TEMP'
resther_TEMPPWF = GetOpacityTransferFunction('resther_TEMP')
resther_TEMPPWF.RescaleTransferFunction(CBMin, CBMax)

# current camera placement for renderView1
camangle = 20
camradius = 0.025

renderView1.CameraPosition = [0, -camradius*np.cos(np.radians(camangle)), camradius*np.sin(np.radians(camangle))]
renderView1.CameraFocalPoint = [0.0000,0, 0]
renderView1.CameraViewUp = [0,0,0]
renderView1.CameraParallelScale = 0.000
renderView1.Background = [1,1,1]  ### White Background

animationScene1 = GetAnimationScene()
animationScene1.StartTime = float(Nearest_Tstep)
animationScene1.GoToFirst()

SaveScreenshot(IMAGE_DIR + '/Capture.png', renderView1, ImageResolution=[1403, 591], FontScaling='Do not scale fonts')
Hide(thermalrmed, renderView1)

clip1 = Clip(Input=thermalrmed)
clip1.ClipType.Normal = [0.0, 1.0, 0.0]

clip1Display = Show(clip1, renderView1)
resther_TEMPLUT.RescaleTransferFunction(CBMin, CBMax)
resther_TEMPPWF.RescaleTransferFunction(CBMin, CBMax)
ColorBar = GetScalarBar(resther_TEMPLUT, renderView1)
ColorBar.Visibility = 1

SaveScreenshot(IMAGE_DIR + '/ClipCapture.png', renderView1, ImageResolution=[1403, 591], FontScaling='Do not scale fonts')
Hide(clip1, renderView1)

calculator1 = Calculator(Input=thermalrmed)
calculator1.ResultArrayName = 'Resulti'
calculator1.Function = 'coords.iHat'
calculator2 = Calculator(Input=calculator1)
calculator2.ResultArrayName = 'Resultj'
calculator2.Function = 'coords.jHat'
calculator3 = Calculator(Input=calculator2)
calculator3.ResultArrayName = 'Resultk'
calculator3.Function = 'coords.kHat'

threshold1 = Threshold(Input=calculator3)
threshold1Display = Show(threshold1, renderView1)
threshold1Display.Representation = 'Surface With Edges'
threshold1.Scalars = ['POINTS', 'Resultj']
threshold1.ThresholdRange = [0, 0.0063]
threshold1Display.EdgeColor = [0.0, 0.0, 0.0]
threshold1Display.DiffuseColor = [0.2, 0.75, 0.996078431372549]

renderView1.CameraPosition = [0, -0.015, 0.00125]
renderView1.CameraFocalPoint = [0, 0, 0.00125]
renderView1.CameraViewUp = [0.0, 0.0, 1.0]

SaveScreenshot(IMAGE_DIR + '/MeshClip.png', renderView1, ImageResolution=[1403, 591], FontScaling='Do not scale fonts')

RenameSource('Thermal', thermalrmed)


if salome.sg.hasDesktop():
  salome.sg.updateObjBrowser(True)
