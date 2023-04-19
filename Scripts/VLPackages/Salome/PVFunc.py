
import numpy as np
import pvsimple

def Camera(FocalPoint,alpha1,alpha2,radius,ViewUp = [0,0,1]):
    renderview = pvsimple.GetActiveViewOrCreate('RenderView')
    CameraPosition = [FocalPoint[0] + radius*np.cos(np.deg2rad(alpha2))*np.sin(np.deg2rad(alpha1)),
                      FocalPoint[1] - radius*np.cos(np.deg2rad(alpha2))*np.cos(np.deg2rad(alpha1)),
                      FocalPoint[2] + radius*np.sin(np.deg2rad(alpha2))]

    renderview.CameraPosition = CameraPosition
    renderview.CameraFocalPoint = FocalPoint
    renderview.CameraViewUp = ViewUp

    return renderview

def GetDataRange(input,resname,restype):
    if restype.lower()=='point':
        data = input.PointData

    res_available = list(data.keys())
    if resname not in res_available:
        print("Warning: {} not in the list of available results")
        return

    ix = res_available.index(resname)
    range = data.GetArray(ix).GetRange()
    return range

def DataRange(input,resname):
    return GetDataRange(input,resname,'point')

