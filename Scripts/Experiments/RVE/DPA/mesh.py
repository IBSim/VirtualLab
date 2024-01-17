import numpy as np
from scipy.interpolate import make_interp_spline, BSpline
from scipy.interpolate import interp1d
import os
import matplotlib.pyplot as plt
import sys



def dpa_calculation(VL,DPADict):
	
	f = open('{}/E/E_0.txt'.format(DPADict['CALC_DIR']),"r") 
	lines=f.readlines()
	yr = [(line.strip().split())for line in lines]
	e=len(yr)
	f.close()
	print(e)
	with open('{}/Rhenium.txt'.format(DPADict['CALC_DIR']),'w') as fout:
	     for i in range(0,e-1):
	         fout.write(str(yr[i][1])+' '+str(yr[i][2])+' '+str(yr[i][3])+' '+str(yr[i][4])+ '\n')
	fout.close()
	with open('{}/Osmium.txt	'.format(DPADict['CALC_DIR']),'w') as fout:
	     for i in range(1,e):
	         fout.write(str(yr[i][1])+' '+str(yr[i][2])+' '+str(yr[i][3])+' '+str(yr[i][4])+ '\n')
	fout.close()
