#
import h5py
import numpy as np
import os
import sys
from Scripts.Common.tools import MeshInfo    
from VLFunctions import MaterialProperty
from Scripts.Common.VLPackages import SalomeRun
from types import SimpleNamespace as Namespace
from importlib import import_module

def Single(VL,DADict):
    print('DADict: ', DADict)
    ParametersMesh_Temp = VL.CreateParameters(VL.Parameters_Master, VL.Parameters_Var,'Mesh')
    ParametersSim_Temp = VL.CreateParameters(VL.Parameters_Master, VL.Parameters_Var,'Sim')
    ParametersDA_Temp = VL.CreateParameters(VL.Parameters_Master, VL.Parameters_Var,'DA')

    DA_Key = list(ParametersDA_Temp.keys())[0]
    ParametersDA = ParametersDA_Temp[DA_Key]

    Mesh_Key = list(ParametersMesh_Temp.keys())[0]
    ParametersMesh = ParametersMesh_Temp[Mesh_Key]

    Sim_Key = list(ParametersSim_Temp.keys())[0]
    ParametersSim = ParametersSim_Temp[Sim_Key]

    ResData = {} # empty dict for result data

#============================================================================
    print('ParametersDA[test]')
    print(ParametersDA)
    print('ParametersMesh[test]')
    print(ParametersMesh)
    print('ParametersSim[test]')
    print(ParametersSim)
#============================================================================
    # results requested
    ResultRelative = ParametersDA['Scalar']
    ResultPlot= ParametersDA['Plot']
    ResultImage= ParametersDA['ScalarImage']

#============================================================================
    # result directory
    ResDir = "{}/{}".format(VL.PROJECT_DIR, ParametersSim['Name'])#, 'Aster') # where we keep result file .rmed
    ResFile =  '{}/Aster/TensileTest.rmed'.format(ResDir)                #os.listdir(ResDir): listing files in directory
    # image directory in results
    ImageDir = "{}/PostAster/ImageDir".format(ResDir)

#=======================Compute the coordinates of the points to be tracked==
    if len(ParametersDA['Distance'])!=0:
        PointCoord = []
        for i in range (len(ParametersDA['Distance'])):
            PointLeft = ([ParametersMesh['HandleLength'] + ParametersMesh['TransRad'] + ParametersMesh['GaugeLength']*0.5 - (ParametersDA['Distance'])[i], ParametersMesh['HandleWidth']*0.5 , ParametersMesh['Thickness']])
            PointRight = ([ParametersMesh['HandleLength']  + ParametersMesh['TransRad'] + ParametersMesh['GaugeLength']*0.5 + (ParametersDA['Distance'])[i], ParametersMesh['HandleWidth']*0.5 , ParametersMesh['Thickness']])
            PointCoord.append([PointLeft, PointRight])   
#============================================================================    
    
    # open result file using h5py
    g = h5py.File(ResFile, 'r')    
    gRes = []
    gRes_track= []
    if ParametersDA['Control'] == 'displacement_control':
        for i in range(len(ResultImage)):
            if ResultImage[i] =='Displacement':
                gRes.append(g['/CHA/Disp_Displacement'])
            if ResultImage[i] =='Strain':
                gRes.append(g['/CHA/Disp_Strain']) 
            if ResultImage[i] =='Stress':
                gRes.append(g['/CHA/Disp_Stress'])
        for i in range(len(ResultRelative)):
            if ResultRelative[i] =='Displacement':
                gRes_track.append(g['/CHA/Disp_Displacement'])
            if ResultRelative[i] =='Strain':
                gRes_track.append(g['/CHA/Disp_Strain']) 
            if ResultRelative[i] =='Stress':
                gRes_track.append(g['/CHA/Disp_Stress'])        
    elif ParametersDA['Control'] == 'force_control':
        for i in range(len(ResultImage)):
            if ResultImage[i] =='Displacement':
                gRes.append(g['/CHA/Force_Displacement'])
            if ResultImage[i] =='Strain':
                gRes.append(g['/CHA/Force_Strain']) 
            if ResultImage[i] =='Stress':
                gRes.append(g['/CHA/Force_Stress'])
        for i in range(len(ResultRelative)):
            if ResultRelative[i] =='Displacement':
                gRes_track.append(g['/CHA/Force_Displacement'])
            if ResultRelative[i] =='Strain':
                gRes_track.append(g['/CHA/Force_Strain']) 
            if ResultRelative[i] =='Stress':
                gRes_track.append(g['/CHA/Force_Stress'])
    else:
        print('***error in DA.Control keyword or output parameters Displacement/Stress/Strain!***')

    if len(ResultImage)!=0:    
        print('Creating images for scalar distribution over top surface using TensileScalarDistribution.py') 
        
    for i in range(len(ResultImage)):
        gRes_temp = gRes[i]
        steps = list(gRes_temp.keys()) # temp values only in last step
        time = [gRes_temp[step].attrs['PDT'] for step in steps]
        
        cstep = []
        ctime = []     
        
        if ParametersDA['CaptureTime'] == 'all': # finding the list of steps for all time increments
            iterator = len(steps)
            for time1, step1 in zip(time, steps):
                ctime.append(time1)
                cstep.append(step1)
        else: # finding a single step for a specific increment assigned in input file
            iterator = 1
            for time1, step1 in zip(time, steps):
                if time1 == ParametersDA['CaptureTime'] :
                    cstep.append(step1)
                    ctime.append(time1)
            
        ResData = {}
        for g in range(iterator): # time loop using the list of step or single step
            GlobalRange = [np.inf, -np.inf]
            ScalarNodes = gRes_temp['{}/NOE/MED_NO_PROFILE_INTERNAL/CO'.format(cstep[g])][:]
            GlobalRange =  [min(min(ScalarNodes),GlobalRange[0]),max(max(ScalarNodes),GlobalRange[1])]
        
            ResData = {'File':ResFile,
                        'Time':ctime[g],
                        'ImageDir':ImageDir,
                        'ScalarImage': ResultImage[i]}
            DADict['ResData'] = ResData
            DADict['GlobalRange'] = GlobalRange
            DADict['ParametersMesh'] = ParametersMesh
            

            GUI = getattr(ParametersDA, 'PVGUI', True)
            ParaVisFile = "{}/TensileScalarDistribution.py".format(os.path.dirname(os.path.abspath(__file__)))
            RC = SalomeRun(ParaVisFile, DataDict=DADict, GUI=GUI)
            if RC:
                return "Error in Salome run while in TensileScalarDistribution.py"
            
    if len(ResultRelative)!=0:    
        print('Creating images for relative displacements over the time between selected points at top surface using TensilePointTracking.py') 
        
    for i in range(len(ResultRelative)):
        gRes_temp = gRes_track[i]
        steps = list(gRes_temp.keys()) # temp values only in last step
        time = [gRes_temp[step].attrs['PDT'] for step in steps]
        
        cstep = []
        ctime = []     
        
        if ParametersDA['CaptureTime'] == 'all': # finding the list of steps for all time increments
            iterator = len(steps)
            for time, step in zip(time, steps):
                ctime.append(time)
                cstep.append(step)
        else: # finding a single step for a specific increment assigned in input file
            iterator = 1
            for time, step in zip(time, steps):
                if time1 == ParametersDA['CaptureTime'] :
                    cstep.append(step)
                    ctime.append(time)
            
        ResData = {}
        for g in range(iterator): # time loop using the list of step or single step
            GlobalRange = [np.inf, -np.inf]
            ScalarNodes = gRes_temp['{}/NOE/MED_NO_PROFILE_INTERNAL/CO'.format(cstep[g])][:]
            GlobalRange =  [min(min(ScalarNodes),GlobalRange[0]),max(max(ScalarNodes),GlobalRange[1])]
            print('GlobalRange',GlobalRange )
            ResData = {'File':ResFile,
                        'Time':ctime[g],
                        'ImageDir':ImageDir,
                        'ResultRelative': ResultRelative[i],
                        'PointCoord': PointCoord}
            DADict['ResData'] = ResData
            DADict['GlobalRange'] = GlobalRange
            DADict['ParametersMesh'] = ParametersMesh
            

            GUI = getattr(ParametersDA, 'PVGUI', True)
            ParaVisFile = "{}/TensilePointTracking.py".format(os.path.dirname(os.path.abspath(__file__)))
            RC = SalomeRun(ParaVisFile, DataDict=DADict, GUI=GUI)
            if RC:
                return "Error in Salome run while in TensilePointTracking.py"            
        
# call image generator script!!
