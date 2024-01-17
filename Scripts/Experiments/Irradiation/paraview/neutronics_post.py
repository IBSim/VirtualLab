#!/bin/bash\n
import sys
import os
import paraview
from paraview.simple import *

def simulation_paraview(CALC_DIR,Name,Headless=False,_Name=None):
               
		# Read the VTK files
		heating_openmc_meshvtk = LegacyVTKReader(FileNames=[f"{CALC_DIR}/heating_openmc_mesh.vtk"])
		damagenergy_openmc_meshvtk = LegacyVTKReader(FileNames=[f"{CALC_DIR}/damage_energy_openmc_mesh.vtk"])
		transform1 = Transform(Input=heating_openmc_meshvtk)
		transform2 = Transform(Input=damagenergy_openmc_meshvtk)
		transform1.Transform = 'Transform'
		transform2.Transform = 'Transform'
		# Convert cm to mm
		transform1.Transform.Scale = [10.0, 10.0, 10.0]
		transform2.Transform.Scale = [10.0, 10.0, 10.0]
		# Obtain the cell centers and save in VTK format
		cellCenters1 = CellCenters(Input=transform1)
		cellCenters2 = CellCenters(Input=transform2)
		SaveData(f"{CALC_DIR}/heating_openmc_mesh_pv.vtk", proxy=transform1,FileType='Ascii')
		SaveData(f"{CALC_DIR}/damage_openmc_mesh_pv.vtk", proxy=transform2,FileType='Ascii')
		SaveData(f"{CALC_DIR}/damage_energy_openmc.vtk", proxy=cellCenters2,FileType='Ascii')
		SaveData(f"{CALC_DIR}/heating_openmc.vtk", proxy=cellCenters1, FileType='Ascii')
