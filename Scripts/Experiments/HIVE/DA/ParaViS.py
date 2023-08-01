#!/usr/bin/env python
'''
Script containing functions used in paravis to visualise results.
'''

import pvsimple
import PVFunc

# camera position to look down on top of the component
aerial_camera= [[0.0225,0.0275,0.0225], # camera focal point
                90, # angle from +ve x direction in xy plane
                90, # angle between xy plane and z axis
                0.12 # distance from focal point
                ] 

JouleCB = {**PVFunc.colorbar_default,'Title':'Joule heating\n(W/m^3)','ComponentTitle':'','RangeLabelFormat' : '%-#.3e'}
JouleTF = {**PVFunc.transferfunc_default,'ApplyPreset':['PuRd', True]}

DiffTF = {**PVFunc.transferfunc_default,'ApplyPreset':['Linear Green (Gr4L)', True]}

def HeatingProfileCompare(res_file,result_names,filenames_compare,filename_diff):
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


if __name__=='__main__':
    EvalList = PVFunc.GetEvalInfo()
    PVFunc.FuncEval(EvalList,globals())


