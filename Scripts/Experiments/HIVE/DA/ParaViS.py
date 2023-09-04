#!/usr/bin/env python
'''
Script containing functions used in paravis to visualise results.
'''

import pvsimple
import PVFunc


camera_default = [[0.0275,0.0225,0.0225], # camera focal point
                40, # alpha1
                30, # alpha2
                0.25 # radius
                ]


DiffTF = {**PVFunc.transferfunc_default,'ApplyPreset':['Linear Green (Gr4L)', True]}

TempCB = {**PVFunc.colorbar_default,'Title':'Temperature (C)'}
TempTF = {**PVFunc.transferfunc_default,'ApplyPreset':['RdOrYl', True]}

VonMisesCB = {**PVFunc.colorbar_default,'Title':'VonMises (MPa)','ComponentTitle':''}
VonMisesTF = {**PVFunc.transferfunc_default,'ApplyPreset':['BuGn', True]}

def HeatingProfileCompare(res_file,result_names,filenames_compare,filename_diff):
    # camera settings for images
    aerial_camera= [[0.0225,0.0275,0.0225], # camera focal point
                    90, # angle from +ve x direction in xy plane
                    90, # angle between xy plane and z axis
                    0.12 # distance from focal point
                    ] 

    # colour bar and transfer function parameters
    JouleCB = {**PVFunc.colorbar_default,'Title':'Joule heating\n(W/m^3)','ComponentTitle':'','RangeLabelFormat' : '%-#.3e'}
    JouleTF = {**PVFunc.transferfunc_default,'ApplyPreset':['PuRd', True]}

    medfile = PVFunc.OpenMED(res_file)
    extractGroup1 = pvsimple.ExtractGroup(Input=medfile)
    extractGroup1.AllGroups = ['GRP_CoilFace']
    # Create plot of the ML model and simulation result with the same colour bar (which is taken from the data at index 1)
    PVFunc.Compare(extractGroup1,result_names,filenames_compare,
                   camera=aerial_camera,CB=JouleCB,TF=JouleTF,compare_ix=1)

    CB = {**JouleCB,'Title':'\u0394 {}'.format(JouleCB['Title'])}
    PVFunc.Difference(extractGroup1,result_names,filename_diff,
                              camera=aerial_camera, CB=CB, TF=DiffTF,
                              absolute_difference=True)



def TemperatureCompare(res_file,result_names,filenames_compare,filename_diff):

    res_obj = PVFunc.OpenMED(res_file)

    # Create plot of the ML model and simulation result with the same colour bar (which is taken from the data at index 1)
    PVFunc.Compare(res_obj,result_names,filenames_compare,
                   camera=camera_default,CB=TempCB,TF=TempTF,compare_ix=1)

    CB = {**TempCB,'Title':'\u0394 {}'.format(TempCB['Title'])}
    PVFunc.Difference(res_obj,result_names,filename_diff,
                              camera=camera_default, CB=CB, TF=DiffTF,
                              absolute_difference=True)

def VonMisesCompare(res_file,result_names,filenames_compare,filename_diff):

    res_obj = PVFunc.OpenMED(res_file)

    # Create plot of the ML model and simulation result with the same colour bar (which is taken from the data at index 1)
    PVFunc.Compare(res_obj,result_names,filenames_compare,
                   camera=camera_default,CB=VonMisesCB,TF=VonMisesTF,compare_ix=1)

    CB = {**VonMisesCB,'Title':'\u0394 {}'.format(VonMisesCB['Title'])}
    PVFunc.Difference(res_obj,result_names,filename_diff,
                              camera=camera_default, CB=CB, TF=DiffTF,
                              absolute_difference=True)


def CaptureTemperature(res_file,result_names,filenames):
    res_obj = PVFunc.OpenMED(res_file)
    PVFunc.ImageCapture(res_obj,result_names,filenames,CB=TempCB,TF=TempTF)

def CaptureVonMises(res_file,result_names,filenames):
    res_obj = PVFunc.OpenMED(res_file)
    PVFunc.ImageCapture(res_obj,result_names,filenames,CB=VonMisesCB,TF=VonMisesTF)


def PlotTC(mesh_file,points,image_dir,radius=0.002):
    pvsimple.HideAll()

    res_obj = PVFunc.OpenMED(mesh_file)
    renderView1 = PVFunc.GetRenderView(camera_default)
    # renderView1.OrientationAxesVisibility = 0

    meddisplay = pvsimple.Show(res_obj, renderView1)
    meddisplay.Representation = 'Surface'
    meddisplay.DiffuseColor = [0.5529411764705883, 0.5529411764705883, 0.5529411764705883] # colour the mesh

    #===============================================================================
    # Add spheres at thermocouple locations
    for c in points:
        # create a new 'Sphere'
        sphere1 = pvsimple.Sphere(Center = c,Radius = radius)
        # set active source
        pvsimple.SetActiveSource(sphere1)
        # show data in view
        sphere1Display = pvsimple.Show(sphere1, renderView1)
        sphere1Display.DiffuseColor = [0.6666666666666666, 0.3333333333333333, 0.0] # colour the spheres

    pic1 = camera_default
    pic2 = camera_default.copy()
    pic2[1] = pic2[1] + 180
    pic3 = [camera_default[0],0,-90,0.15] 

    for i,pic in enumerate([pic1,pic2,pic3]):
        filename = "{}/Angle_{}.png".format(image_dir,i+1)
        renderView1 = PVFunc.UpdateCamera(renderView1,*pic)
        pvsimple.SaveScreenshot(filename, renderView1,**PVFunc.screenshot_default)




if __name__=='__main__':
    EvalList = PVFunc.GetEvalInfo()
    PVFunc.FuncEval(EvalList,globals())


