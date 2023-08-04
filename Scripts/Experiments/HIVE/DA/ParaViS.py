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

VonMisesCB = {**PVFunc.colorbar_default,'Title':'VonMises (Pa)','ComponentTitle':'','RangeLabelFormat' : '%-#.3e'}
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


if __name__=='__main__':
    EvalList = PVFunc.GetEvalInfo()
    PVFunc.FuncEval(EvalList,globals())


