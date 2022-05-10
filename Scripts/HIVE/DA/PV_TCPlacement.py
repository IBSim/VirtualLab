import sys
import os
import salome
import SalomeFunc

import pvsimple
pvsimple.ShowParaviewView()

# This function gives the ArgDict dictionary we passed to SalomeRun
DataDict = SalomeFunc.GetArgs()


# create a new 'MED Reader'
Resrmed = pvsimple.MEDReader(FileName=DataDict['File'])

# set active source
pvsimple.SetActiveSource(Resrmed)

# get active view
renderView1 = pvsimple.GetActiveViewOrCreate('RenderView')

mL_resrmedDisplay = pvsimple.Show(Resrmed, renderView1)
mL_resrmedDisplay.Representation = 'Surface'

renderView1.ResetCamera()
cam = pvsimple.GetActiveCamera() # Get camera properties

#===============================================================================
# Add spheres at thermocouple locations
Radius = DataDict.get('Radius',0.001)
for c in DataDict['Centres']:
    # create a new 'Sphere'
    sphere1 = pvsimple.Sphere(Center = c,Radius = Radius)
    # set active source
    pvsimple.SetActiveSource(sphere1)
    # show data in view
    sphere1Display = pvsimple.Show(sphere1, renderView1)
    sphere1Display.DiffuseColor = [0.6666666666666666, 0.3333333333333333, 0.0]

#===============================================================================
# Create orthogonal images of component to show thermocouple locations
FP = list(cam.GetFocalPoint()) #focal point
for i,comp in enumerate(['x','y','z']):
    for direction,ext in zip([1,-1],['p','n']):
        _fp = FP.copy()
        _fp[i] = direction
        # renderView1.CameraPosition = _fp
        cam.SetPosition(_fp)
        renderView1.ResetCamera()
        pvsimple.SaveScreenshot('{}/image_{}{}.png'.format(DataDict['OutputDir'],comp,ext),
                                renderView1)
