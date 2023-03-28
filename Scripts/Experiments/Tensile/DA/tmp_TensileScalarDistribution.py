
import sys
import os
sys.dont_write_bytecode=True
import salome
import numpy as np
from importlib import import_module
import SalomeFunc
'''
import SalomePyQt
SalomePyQt.SalomePyQt().activateModule('ParaViS')
'''
import pvsimple

pvsimple.ShowParaviewView()

DADict = SalomeFunc.GetArgs()

ResData = DADict['ResData']

CBmin,CBmax = DADict['GlobalRange']

ParametersMesh = DADict['ParametersMesh']

# get active view
renderView1 = pvsimple.GetActiveViewOrCreate('RenderView') 

# create a new 'MED Reader'
tensileTestrmed = pvsimple.MEDReader(FileName=ResData['File']) 

# show data in view
tensileTestrmedDisplay = pvsimple.Show(tensileTestrmed, renderView1) 

# trace defaults for the display properties.
tensileTestrmedDisplay.Representation = 'Surface' 

# reset view to fit data
renderView1.ResetCamera() 

# update the view to ensure updated data information
renderView1.Update() 

# set scalar coloring
pvsimple.ColorBy(tensileTestrmedDisplay, ('POINTS', ResData['ScalarImage'],'Magnitude' ))

# rescale color and/or opacity maps used to include current data range
tensileTestrmedDisplay.RescaleTransferFunctionToDataRange(True, False) 

tensileTestrmedDisplay.SetScalarBarVisibility(renderView1, True) # show color bar/color legend

# get color transfer function/color map for the output parameter
resther_LUT = pvsimple.GetColorTransferFunction(ResData['ScalarImage'])

# get opacity transfer function/opacity map for the output parameter
resther_PWF = pvsimple.GetOpacityTransferFunction(ResData['ScalarImage'])

# Properties modified on renderView1 (white background)
renderView1.Background = [1,1,1] 

# rescale color and/or opacity maps used to exactly fit the current
tensileTestrmedDisplay.RescaleTransferFunctionToDataRange(False, True) 

# Rescale transfer function
resther_LUT.RescaleTransferFunction(0.0, 1.0)

# Rescale transfer function
resther_PWF.RescaleTransferFunction(0.0, 1.0)

resther_LUTColorBar = pvsimple.GetScalarBar(resther_LUT, renderView1)
# Properties modified on disp_StrainLUTColorBar
BarLength = 0.25
FontSize = 12
resther_LUTColorBar.AutoOrient = 0
resther_LUTColorBar.Orientation = 'Horizontal'
resther_LUTColorBar.WindowLocation = 'LowerCenter'
resther_LUTColorBar.HorizontalTitle = 1

resther_LUTColorBar.ScalarBarLength = BarLength
resther_LUTColorBar.ScalarBarThickness = FontSize
resther_LUTColorBar.Title = ResData['ScalarImage']
resther_LUTColorBar.TitleFontSize = FontSize
resther_LUTColorBar.CustomLabels = list(np.round(np.linspace(CBmin, CBmax,7), 2))
resther_LUTColorBar.UseCustomLabels = 1
resther_LUTColorBar.AddRangeLabels = 1

# format in SI unit system
if  ResData['ScalarImage'] == 'Stress':
    resther_LUTColorBar.RangeLabelFormat = '%-#.2e'  
else:
    resther_LUTColorBar.RangeLabelFormat = '%-#.2f' 


resther_LUTColorBar.Visibility = 1


resther_LUT = pvsimple.GetColorTransferFunction(ResData['ScalarImage'])
resther_LUT.RescaleTransferFunction(CBmin, CBmax)

resther_PWF = pvsimple.GetOpacityTransferFunction(ResData['ScalarImage'])
resther_PWF.RescaleTransferFunction(CBmin, CBmax)

renderView1.Update()
# Properties modified on LUTColorBar
resther_LUTColorBar.TitleColor = [0.0, 0.0, 0.0] 
resther_LUTColorBar.LabelColor = [0.0, 0.0, 0.0]

# Properties modified on LUTColorBar
resther_LUTColorBar.TitleBold = 1
resther_LUTColorBar.LabelBold = 1

# Properties modified on renderView1
renderView1.OrientationAxesVisibility = 1
# current camera placement for renderView1
renderView1.CameraPosition = [0.054392304845413256, 0.012, 0.14854421498295267]
renderView1.CameraFocalPoint = [0.054392304845413256, 0.012, 0.0015]
renderView1.CameraParallelScale = 0.03805784330973757
renderView1.CameraParallelProjection = 1
renderView1.OrientationAxesLabelColor = [0.0, 0.0, 0.0]
renderView1.OrientationAxesOutlineColor = [0.0, 0.0, 0.0] #for rotation axis colour
##
animationScene1 = pvsimple.GetAnimationScene()
#*****************************************************************************#
if ResData['Time'] > animationScene1.EndTime:
    animationScene1.AnimationTime = animationScene1.EndTime
else :
    animationScene1.AnimationTime = ResData['Time']

outfile = "{}/{}_time_{}.png".format(ResData['ImageDir'], ResData['ScalarImage'], ResData['Time'])
if not os.path.exists(ResData['ImageDir']):
    os.makedirs(ResData['ImageDir'])
pvsimple.SaveScreenshot(outfile, renderView1, ImageResolution=[1400, 590], FontScaling='Do not scale fonts')
pvsimple.Hide(tensileTestrmed, renderView1)
print("Created image ", outfile)
#*****************************************************************************#

for source in pvsimple.GetSources().values():
    pvsimple.Delete(source)
