#!/usr/bin/env python
import sys
import os
import json
import csv
import numpy as np
import paraview
from paraview.simple import *
import paraview.servermanager
def lifecycle_paraview(CALC_DIR,Name,Headless=False,_Name=None):

    fpath = "{CALC_DIR}/Aster/vmis/"
    fname = 'vmis_0_0.vtu'
    case = 'test'
    
    conditions = ["TMax_tn", "VMMax_tn", "TMin_tn"]
               
    for condition in conditions:            
                 # disable automatic camera reset on 'Show'
        paraview.simple._DisableFirstRenderCameraReset()
  
        file = XMLUnstructuredGridReader(registrationName='vmis_0_0.vtu',FileName=[f"{CALC_DIR}/Aster/vmis/vmis_0_0.vtu"])
        file.PointArrayStatus = ['P1______SIEQ_NOEU','rth_____TEMP']

           # get active view
        renderView1 = GetActiveViewOrCreate('RenderView')
    
    	# show data in view
        fileDisplay = Show(file, renderView1, 'UnstructuredGridRepresentation')
    	# get color transfer function/color map for 'Temperature'
        rth_____TEMPLUT = GetColorTransferFunction('rth_____TEMP')
    
    	# get opacity transfer function/opacity map for 'Temperature'
        rth_____TEMPPWF = GetOpacityTransferFunction('rth_____TEMP')
    
    	# reset view to fit data
        renderView1.ResetCamera(False)
    
    	# get the material library
        materialLibrary1 = GetMaterialLibrary()
    
    	# show color bar/color legend
        fileDisplay.SetScalarBarVisibility(renderView1, True)
    
    	# update the view to ensure updated data information
        renderView1.Update()

        bound = file.GetDataInformation().GetBounds()
    
        centre = [(bound[0] + bound[1])/2, (bound[2] + bound[3])/2, 
              (bound[4] + bound[5])/2]
    
        radius = abs(centre[0] - bound[0])
        print (radius)
        if condition == "VMMax_tn": 
           QueryString = "(P1______SIEQ_NOEU == max(P1______SIEQ_NOEU))"
        if condition == "TMax_tn": 
           QueryString = '(rth_____TEMP == max(rth_____TEMP))'
        if condition == "TMin_tn": 
           QueryString = "(rth_____TEMP == min(rth_____TEMP))"
    
  
        print("Query =", QueryString + ":")

        QuerySelect(QueryString, FieldType='POINT', InsideOut=0)

        extractSelection1 = ExtractSelection(registrationName='ExtractSelection1', Input=file)
    
        renderView1 = GetActiveViewOrCreate('RenderView')
 
        layout1 = GetLayout()   
   
    
        # split cell
        layout1.SplitVertical(0, 0.5)
    
    	# set active view
        SetActiveView(None)
    
        # Create a new 'SpreadSheet View'
        spreadSheetView1 = CreateView('SpreadSheetView')
        spreadSheetView1.ColumnToSort = ''
        spreadSheetView1.BlockSize = 1024
    
    	# assign view to a particular cell in the layout
        AssignViewToLayout(view=spreadSheetView1, layout=layout1, hint=0)
    
    	# find source
        file = FindSource(fname)
    
        # show data in view
        extractSelection1Display = Show(extractSelection1, spreadSheetView1, 
                                    'SpreadSheetRepresentation')
    
        # update the view to ensure updated data information
        spreadSheetView1.Update()
    
        selection_file = f"{CALC_DIR}/Aster/" + 'selection_test_{}.csv'.format(condition)
    	# export view
        ExportView(selection_file, view=spreadSheetView1, RealNumberPrecision=24)
    
        # Extracts the coordinates of the specific point
        with open(selection_file) as csvfile:
             csv_reader=csv.DictReader(csvfile)
             for row in csv_reader:
                
                
                 coordinates = [float(row['Points_0']), 
                     float(row['Points_1']), 
                     float(row['Points_2'])]

        SetActiveSource(file)
    	# create a new 'Plot Over Line'
        plotOverLine1 = PlotOverLine(registrationName='PlotOverLine1', 
                                 Input=file)
        print(row['Points_2'])
    	# Starting point of plot over line, centre along x and z
        plotOverLine1.Point1 = [centre[0], coordinates[1], centre[2]]
    
    	# Projecting point onto outer radius
    
        dx = coordinates[0] - centre[0]
        dz = coordinates[2] - centre[2]
    
        if dx > 0:
           theta = np.arctan(abs(dz/dx))
        elif dz < 0:
           theta = np.pi + np.arctan(abs(dz/dx))
        else: 
           theta = np.pi - np.arctan(abs(dz/dx))
    
    
        proj_x = centre[0] + radius*np.cos(theta)
        proj_z = centre[2] + radius*np.sin(theta)
    
        projection = [proj_x, coordinates[1], proj_z]
    
        plotOverLine1.Point2 = projection
    
        print("\t Coordinates of centre: ", centre)
        print("\t Coordinates of query point: ", coordinates)
        print("\t Coordinates of projected point: ", projection)
    
    	# Create a new 'SpreadSheet View'
        spreadSheetView2 = CreateView('SpreadSheetView')
        spreadSheetView2.ColumnToSort = ''
        spreadSheetView2.BlockSize = 1024
    
    	# show data in view
        plotOverLine1Display = Show(plotOverLine1, spreadSheetView2, 
                                'SpreadSheetRepresentation')
    
    	# Create a new 'Line Chart View'
        lineChartView1 = CreateView('XYChartView')
    
    	# show data in view
        plotOverLine1Display_1 = Show(plotOverLine1, lineChartView1, 
                                  'XYChartRepresentation')
    
    	# add view to a layout so it's visible in UI
        AssignViewToLayout(view=lineChartView1, layout=layout1, hint=0)
    
    	# set active view
        SetActiveView(spreadSheetView2)
        print(spreadSheetView2)

    	# Properties modified on plotOverLine1Display
        plotOverLine1Display.Assembly = ''
    
    	# export view
        ExportView(f"{CALC_DIR}/Aster/" + case + '_' + '{}.csv'.format(condition), 
               view=spreadSheetView2, RealNumberPrecision=24)
    
    	# removing the CSV files created for the query selection
        os.remove(selection_file)
        Disconnect()
        Connect()
    fpath = "{CALC_DIR}/Aster/vmis/"
    fname = 'vmis_0_0.vtu'
    case = 'test'
    
    
    fpath1 = "{CALC_DIR}/Aster/yield/"
    fname1 = 'yield_0_0.vtu'
    case1 = 'yield'

    conditions = ["TMax_tn", "VMMax_tn", "TMin_tn"]
    
    for condition in conditions:
     # disable automatic camera reset on 'Show'
        paraview.simple._DisableFirstRenderCameraReset()
  
        file = XMLUnstructuredGridReader(registrationName='vmis_0_0.vtu',FileName=[f"{CALC_DIR}/Aster/vmis/vmis_0_0.vtu"])
        file.PointArrayStatus = ['P1______SIEQ_NOEU','rth_____TEMP']
    
    # get active view
        renderView1 = GetActiveViewOrCreate('RenderView')
    
    # show data in view
        fileDisplay = Show(file, renderView1, 'UnstructuredGridRepresentation')
    # get color transfer function/color map for 'Temperature'
        rth_____TEMPLUT = GetColorTransferFunction('rth_____TEMP')
    
    # get opacity transfer function/opacity map for 'Temperature'
        rth_____TEMPPWF = GetOpacityTransferFunction('rth_____TEMP')
    
    # reset view to fit data
        renderView1.ResetCamera(False)
    
    # get the material library
        materialLibrary1 = GetMaterialLibrary()
    
    # show color bar/color legend
        fileDisplay.SetScalarBarVisibility(renderView1, True)
    
    # update the view to ensure updated data information
        renderView1.Update()

        bound = file.GetDataInformation().GetBounds()
    
        centre = [(bound[0] + bound[1])/2, (bound[2] + bound[3])/2, 
              (bound[4] + bound[5])/2]
    
        radius = abs(centre[0] - bound[0])
    
        if condition == "VMMax_tn": 
           QueryString = "(P1______SIEQ_NOEU == max(P1______SIEQ_NOEU))"
        if condition == "TMax_tn": 
           QueryString = '(rth_____TEMP == max(rth_____TEMP))'
        if condition == "TMin_tn": 
           QueryString = "(rth_____TEMP == min(rth_____TEMP))"
    
  
        print("Query =", QueryString + ":")

        QuerySelect(QueryString, FieldType='POINT', InsideOut=0)

        extractSelection1 = ExtractSelection(registrationName='ExtractSelection1', Input=file)
    
        renderView1 = GetActiveViewOrCreate('RenderView')
 
        layout1 = GetLayout()   
   
    
    # split cell
        layout1.SplitVertical(0, 0.5)
    
    # set active view
        SetActiveView(None)
    
    # Create a new 'SpreadSheet View'
        spreadSheetView1 = CreateView('SpreadSheetView')
        spreadSheetView1.ColumnToSort = ''
        spreadSheetView1.BlockSize = 1024
    
    # assign view to a particular cell in the layout
        AssignViewToLayout(view=spreadSheetView1, layout=layout1, hint=0)
    
    # find source
        file = FindSource(fname)
    
    # show data in view
        extractSelection1Display = Show(extractSelection1, spreadSheetView1, 
                                    'SpreadSheetRepresentation')
    
    # update the view to ensure updated data information
        spreadSheetView1.Update()
    
        selection_file = f"{CALC_DIR}/Aster/" + 'selection_test_{}.csv'.format(condition)
    # export view
        ExportView(selection_file, view=spreadSheetView1, RealNumberPrecision=24)
    
    # Extracts the coordinates of the specific point
     # Extracts the coordinates of the specific point
        with open(selection_file) as csvfile:
             csv_reader=csv.DictReader(csvfile)
             for row in csv_reader:
                 
                 coordinates = [float(row['Points_0']), 
                   float(row['Points_1']), 
                   float(row['Points_2'])]
    

        SetActiveSource(file)
    # create a new 'Plot Over Line'
        plotOverLine1 = PlotOverLine(registrationName='PlotOverLine1', 
                                 Input=file)
 
    # Starting point of plot over line, centre along x and z
        plotOverLine1.Point1 = [centre[0], coordinates[1], centre[2]]
    
    # Projecting point onto outer radius
    
        dx = coordinates[0] - centre[0]
        dz = coordinates[2] - centre[2]
    
        if dx > 0:
          theta = np.arctan(abs(dz/dx))
        elif dz < 0:
          theta = np.pi + np.arctan(abs(dz/dx))
        else: 
          theta = np.pi - np.arctan(abs(dz/dx))
    
    
        proj_x = centre[0] + radius*np.cos(theta)
        proj_z = centre[2] + radius*np.sin(theta)
    
        projection = [proj_x, coordinates[1], proj_z]
    
        plotOverLine1.Point2 = projection
    
        print("\t Coordinates of centre: ", centre)
        print("\t Coordinates of query point: ", coordinates)
        print("\t Coordinates of projected point: ", projection)
    
    # Create a new 'SpreadSheet View'
        spreadSheetView2 = CreateView('SpreadSheetView')
        spreadSheetView2.ColumnToSort = ''
        spreadSheetView2.BlockSize = 1024
    
    # show data in view
        plotOverLine1Display = Show(plotOverLine1, spreadSheetView2, 
                                'SpreadSheetRepresentation')
    
    # Create a new 'Line Chart View'
        lineChartView1 = CreateView('XYChartView')
    
    # show data in view
        plotOverLine1Display_1 = Show(plotOverLine1, lineChartView1, 
                                  'XYChartRepresentation')
    
    # add view to a layout so it's visible in UI
        AssignViewToLayout(view=lineChartView1, layout=layout1, hint=0)
    
    # set active view
        SetActiveView(spreadSheetView2)
        print(spreadSheetView2)

    # Properties modified on plotOverLine1Display
        plotOverLine1Display.Assembly = ''
    
    # export view
        ExportView(f"{CALC_DIR}/Aster/" + case + '_' + '{}.csv'.format(condition), 
               view=spreadSheetView2, RealNumberPrecision=24)
    
    # removing the CSV files created for the query selection
        os.remove(selection_file)


  
  
        file1 = XMLUnstructuredGridReader(registrationName='yield_0_0.vtu',FileName=[f"{CALC_DIR}/Aster/yield/yield_0_0.vtu"])
        file1.PointArrayStatus = ['rth41___FLUX_NOEU']
    
    # get active view
        renderView1 = GetActiveViewOrCreate('RenderView')
    
    # show data in view
        fileDisplay = Show(file, renderView1, 'UnstructuredGridRepresentation')
    # get color transfer function/color map for 'Temperature'
        rth41___FLUX_NOEULUT = GetColorTransferFunction('rth41___FLUX_NOEU')
    
    # get opacity transfer function/opacity map for 'Temperature'
        rth41___FLUX_NOEUPWF = GetOpacityTransferFunction('rth41___FLUX_NOEU')
    
    # reset view to fit data
        renderView1.ResetCamera(False)
    
    # get the material library
        materialLibrary1 = GetMaterialLibrary()
    
    # show color bar/color legend
        fileDisplay.SetScalarBarVisibility(renderView1, True)
    
    # update the view to ensure updated data information
        renderView1.Update()

 
   
    
        QueryString = '(rth41___FLUX_NOEU == (rth41___FLUX_NOEU))'
   
    
  
        print("Query =", QueryString + ":")

        QuerySelect(QueryString, FieldType='POINT', InsideOut=0)

        extractSelection1 = ExtractSelection(registrationName='ExtractSelection1', Input=file1)
    
        renderView1 = GetActiveViewOrCreate('RenderView')
 
        layout1 = GetLayout()   
   
    
      # split cell
        layout1.SplitVertical(0, 0.5)
    
    # set active view
        SetActiveView(None)
    
    # Create a new 'SpreadSheet View'
        spreadSheetView1 = CreateView('SpreadSheetView')
        spreadSheetView1.ColumnToSort = ''
        spreadSheetView1.BlockSize = 1024
    
    # assign view to a particular cell in the layout
        AssignViewToLayout(view=spreadSheetView1, layout=layout1, hint=0)
    
    # find source
        file = FindSource(fname1)
    
    # show data in view
        extractSelection1Display = Show(extractSelection1, spreadSheetView1, 
                                    'SpreadSheetRepresentation')
    
    # update the view to ensure updated data information
        spreadSheetView1.Update()
    
        selection_file1 = f"{CALC_DIR}/Aster/" + 'selection_test1_{}.csv'.format(condition)
    # export view
        ExportView(selection_file1, view=spreadSheetView1, RealNumberPrecision=24)
    
    # Extracts the coordinates of the specific point
        with open(selection_file1) as csvfile:
             csv_reader=csv.DictReader(csvfile)
             for row in csv_reader:
                 
                 coordinates = [float(row['Points_0']), 
                   float(row['Points_1']), 
                   float(row['Points_2'])]

        SetActiveSource(file)
    # create a new 'Plot Over Line'
        plotOverLine1 = PlotOverLine(registrationName='PlotOverLine1', 
                                 Input=file1)
 
    # Starting point of plot over line, centre along x and z
        plotOverLine1.Point1 = [centre[0], coordinates[1], centre[2]]
    
    # Projecting point onto outer radius
  
    
    
        proj_x = centre[0] + radius*np.cos(theta)
        proj_z = centre[2] + radius*np.sin(theta)
    
        projection = [proj_x, coordinates[1], proj_z]
    
        plotOverLine1.Point2 = projection
    
        print("\t Coordinates of centre: ", centre)
        print("\t Coordinates of query point: ", coordinates)
        print("\t Coordinates of projected point: ", projection)
    
    # Create a new 'SpreadSheet View'
        spreadSheetView2 = CreateView('SpreadSheetView')
        spreadSheetView2.ColumnToSort = ''
        spreadSheetView2.BlockSize = 1024
    
    # show data in view
        plotOverLine1Display = Show(plotOverLine1, spreadSheetView2, 
                                'SpreadSheetRepresentation')
    
    # Create a new 'Line Chart View'
        lineChartView1 = CreateView('XYChartView')
    
    # show data in view
        plotOverLine1Display_1 = Show(plotOverLine1, lineChartView1, 
                                  'XYChartRepresentation')
    
    # add view to a layout so it's visible in UI
        AssignViewToLayout(view=lineChartView1, layout=layout1, hint=0)
    
    # set active view
        SetActiveView(spreadSheetView2)
        print(spreadSheetView2)

    # Properties modified on plotOverLine1Display
        plotOverLine1Display.Assembly = ''
    
    # export view
        ExportView(f"{CALC_DIR}/Aster/" + case1 + '_' + '{}.csv'.format(condition), 
               view=spreadSheetView2, RealNumberPrecision=24)
    
    # removing the CSV files created for the query selection
        os.remove(selection_file1)
        Disconnect()
        Connect()
