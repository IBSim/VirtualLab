#!/bin/bash\n
import sys
import os
import paraview
from paraview.simple import *

def lifecycle_paraview(CALC_DIR,Name,Headless=False,_Name=None):
               
		# Read the VTK files
		vmis = MEDReader(FileNames=[f"{CALC_DIR}/Aster/vmis.rmed"])
		
		SaveData(f"{CALC_DIR}/Aster/vmis.vtm", proxy=vmis, ChooseArraysToWrite=1,

    			PointDataArrays=['FamilyIdNode', 'P1______SIEQ_NOEU', 'rth_____TEMP', 'FamilyIdCell', 'NumIdCell'],

    			CellDataArrays=['FamilyIdNode', 'P1______SIEQ_NOEU', 'rth_____TEMP', 'FamilyIdCell', 'NumIdCell'])



