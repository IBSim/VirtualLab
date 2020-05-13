import sys
sys.dont_write_bytecode=True
import salome
import numpy as np

import Parameters
import PathVL

###
### PARAVIS component
###

import pvsimple
pvsimple.ShowParaviewView()
# trace generated using paraview version 5.6.0-RC1

#### import the simple module from the paraview
from pvsimple import *

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
#renderView1.ViewSize = (1400,590)

thermalrmed = MEDReader(FileName="{}/{}.rmed".format(PathVL.ASTER_DIR, Parameters.ResName))

# show data in view
thermalrmedDisplay = Show(thermalrmed, renderView1)
thermalrmedDisplay.Representation = 'Surface'

# current camera placement for renderView1
camangle = 30
camradius = 0.025
renderView1.CameraPosition = [0, -camradius*np.cos(np.radians(camangle)), camradius*np.sin(np.radians(camangle))]
renderView1.CameraFocalPoint = [0,0,-0.002]
renderView1.CameraViewUp = [0,0,0]
renderView1.CameraParallelScale = 0.000
renderView1.Background = [1,1,1]  ### White Background

# set scalar coloring
ColorBy(thermalrmedDisplay, ('POINTS', 'resther_TEMP', 'TEMP'))

Time = 0.008
timesteps = np.array(thermalrmed.TimestepValues)
TimeIx = np.argmin(abs(timesteps - Time))

animationScene1 = GetAnimationScene()
animationScene1.StartTime = timesteps[TimeIx]
animationScene1.GoToFirst()

Temprange = thermalrmed.PointData.GetArray(1).GetRange()
Temprange = [np.floor(Temprange[0]), np.ceil(Temprange[1])]
resther_TEMPLUT = GetColorTransferFunction('resther_TEMP')
resther_TEMPLUT.NumberOfTableValues = 12
resther_TEMPLUT.RescaleTransferFunction(Temprange[0], Temprange[1])
resther_TEMPPWF = GetOpacityTransferFunction('resther_TEMP')
resther_TEMPPWF.RescaleTransferFunction(Temprange[0], Temprange[1])

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
ColorBar.CustomLabels = list(np.round(np.linspace(Temprange[0],Temprange[1],7), 2))
ColorBar.UseCustomLabels = 1
ColorBar.AddRangeLabels = 0

SaveScreenshot("{}/Capture.png".format(PathVL.POST_DIR), renderView1, ImageResolution=[1400, 590], FontScaling='Do not scale fonts')
print("Created image Capture.png\n")
Hide(thermalrmed, renderView1)

clip1 = Clip(Input=thermalrmed)
clip1.ClipType.Normal = [0.0, -1.0, 0.0]
clip1Display = Show(clip1, renderView1)
ColorBar.Visibility = 1
SaveScreenshot("{}/ClipCapture.png".format(PathVL.POST_DIR), renderView1, ImageResolution=[1400, 590], FontScaling='Do not scale fonts')
print("Created image ClipCapture.png\n")
Hide(clip1, renderView1)

thermalrmedDisplay = Show(thermalrmed, renderView1)
ColorBy(thermalrmedDisplay, None)
thermalrmedDisplay.Representation = 'Surface With Edges'
thermalrmedDisplay.EdgeColor = [0.0, 0.0, 0.0]
thermalrmedDisplay.DiffuseColor = [0.2, 0.75, 0.996078431372549]
SaveScreenshot("{}/Mesh.png".format(PathVL.POST_DIR), renderView1, ImageResolution=[1400, 590], FontScaling='Do not scale fonts')
print("Created image Mesh.png\n")
Hide(thermalrmed, renderView1)

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

SaveScreenshot("{}/MeshCrossSection.png".format(PathVL.POST_DIR), renderView1, ImageResolution=[1400, 590], FontScaling='Do not scale fonts')
print("Created image MeshCrossSection.png\n")

RenameSource('Thermal', thermalrmed)




