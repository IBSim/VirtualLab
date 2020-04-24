#!/usr/bin/python

# Add path below to .bash_profile (or similar)
#PYTHONPATH=~/PATH_TO_PARAFEM/parafem/bin

import numpy as np
import math
import sys
import os

# Stop .pyc file being created
sys.dont_write_bytecode=True

def KCalc(ttr2,l,m,vol,Cp):
	#Find T at half of Tequilibrium
	length_ttr2 = int(len(ttr2))
	Tmax=ttr2[length_ttr2-1,1]
	T_h=ttr2[0,1]+((Tmax)-ttr2[0,1])/2
	#Interpolate to get half-rise time
	t_h=np.interp(T_h,ttr2[:,1],ttr2[:,0])

	alpha=(0.1388*(l**2))/t_h
	rho=m/vol
	k=alpha*Cp*rho
	return t_h, k, alpha

def main():
	if len(sys.argv) == 2:
		fname=sys.argv[1]
		module = __import__(fname)

		#Define sample variables
		l=module.l
		rad=module.rad
		m=module.m
		Cp=module.Cp
		vol = np.pi*l*rad**2

		Path = os.path.dirname(os.path.abspath(__file__))
		fname=Path+'/'+fname+'.dat'

		#Open file and read data into array
		ttr2in = np.fromfile(fname,dtype=float,sep=" ")
		length_ttr2in = int(len(ttr2in))
		ttr2=np.zeros(shape=(length_ttr2in/2,2),dtype=float)
		i=0
		for line in range(0,length_ttr2in,2):
		    ttr2[i,0]=ttr2in[line]
		    ttr2[i,1]=ttr2in[line+1]
		    i=i+1
		t_h, k, alpha = KCalc(ttr2,l,m,vol,Cp)
		print('Half time = ',t_h,'m^2/s')
		print('alpha = ',alpha,'m^2/s')
		print('k = ',k,'W/m K')
	else:
		print "Error - Name of data file required as argument"

if __name__ == "__main__":
	main()

