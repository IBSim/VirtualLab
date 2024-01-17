#!/usr/bin/python

import numpy as np

import matplotlib.pyplot as plt

import sys

def dpa_calculation(VL,DPADict):
	ft3 = open("{}/caucy1.txt".format(DPADict['CALC_DIR']),"w")

	f=open('{}/F/F_0.txt'.format(DPADict['CALC_DIR']),"r")
	lines=f.readlines()
	si1xz = [((line.strip().split()))for line in lines]
	po=len(si1xz)
	for i in range(int(po)):
    	    ft3 = open("{}/caucy1.txt".format(DPADict['CALC_DIR']),"a")
    	    ft3.write(str(si1xz[i][83])+" "+str(si1xz[i][92])+"\n")
	ft3.close()  

	ft3 = open("{}/caucy2.txt".format(DPADict['CALC_DIR']),"w")

	f=open('{}/F/F_0.txt'.format(DPADict['CALC_DIR']),"r")
	lines=f.readlines()
	si1xz = [((line.strip().split()))for line in lines]
	po=len(si1xz)
	for i in range(int(po)):
    	    ft3 = open("{}/caucy2.txt".format(DPADict['CALC_DIR']),"a")
    	    ft3.write(str(si1xz[i][83])+" "+str(si1xz[i][92])+"\n")
	ft3.close()  
	ft3 = open("{}/caucy3.txt".format(DPADict['CALC_DIR']),"w")

	f=open('{}/F/F_0.txt'.format(DPADict['CALC_DIR']),"r")
	lines=f.readlines()
	si1xz = [((line.strip().split()))for line in lines]
	po=len(si1xz)
	for i in range(int(po)):
    	    ft3 = open("{}/caucy3.txt".format(DPADict['CALC_DIR']),"a")
    	    ft3.write(str(si1xz[i][83])+" "+str(si1xz[i][92])+"\n")
	ft3.close()    
    
	def line_tuple(filename,cols=(0,1)):
    	    return np.loadtxt(filename,usecols=cols,unpack=True)
	#parse each line from the datafile into a tuple of the form (xvals,yvals)
	#store that tuple in a list.
	data = [line_tuple(fname) for fname in ("{}/caucy1.txt".format(DPADict['CALC_DIR']),"{}/caucy1.txt".format(DPADict['CALC_DIR']),"{}/caucy1.txt".format(DPADict['CALC_DIR']))]
	#This is the minimum and maximum from all the datapoints.
	xmin = min(line[0].min() for line in data)
	xmax = max(line[0].max() for line in data)

	#100 points evenly spaced along the x axis
	x_points = np.linspace(xmin,xmax,1000)

	#interpolate your values to the evenly spaced points.
	interpolated = [np.interp(x_points,d[0],d[1]) for d in data]

	#Now do the averaging.
	averages = [np.average(x) for x in zip(*interpolated)]
	window_len = 3
	kernel = np.ones(window_len, dtype=float)/window_len
	y_smooth = np.convolve(averages, kernel, 'same')
	#put the average value along with it's x point into a file.
	with open("{}/outfile".format(DPADict['CALC_DIR']),'w') as fout:
    	     for x,avg in zip(x_points,averages):
    	         fout.write('{0} {1}\n'.format(x,avg))

	fout.close()

	myList=[]
	myList1=[]
	myListdup1=[]
	myListdup=[]
	f = open("{}/outfile".format(DPADict['CALC_DIR']),"r")
	lines=f.readlines()
	yr = [(line.strip().split())for line in lines]
	e=len(yr)
	f.close()

	for i in range(e):
    	    if float(yr[i][0])>=0 and float(yr[i][1])>=0:
               myList.append(float(yr[i][0]))
               myListdup.append(float(yr[i][1]))


	for i in range(50):
    	    if float(yr[i][0])>=0 and float(yr[i][1])>=0:
               myList1.append(float(yr[i][0]))
               myListdup1.append(float(yr[i][1]))




	k2=[x*161000 for x in myListdup1]

	slope, intercept = np.polyfit(myList1,k2,1)
	print(slope)
	with open("{}/outfile2.txt".format(DPADict['CALC_DIR']),'w') as fout:
             for x,avg in zip(myList,myListdup):
                 fout.write('{0} {1}\n'.format(x,avg*161000))

	fout.close()

	k=[0,.001,.002,.003,.004,.005,.006,.007,.008,.009,.01]
	stress=[]
	for i in range(50):
            stress.append(float((yr[i][0])))
    
	k1=[x*161000 for x in myListdup]


	l1 = [x for x in myList]
	l = [slope*(x-.002) for x in l1]
	L1=np.array(l1)
	L=np.array(l)
	MYLIST=np.array(myList)
	MYLISTDUP=np.array(myListdup)
	with open("{}/outfile1".format(DPADict['CALC_DIR']),'w') as fout:
             for x,avg in zip(myList,myListdup):
                 fout.write('{0} {1}\n'.format(x,avg*161000))

	fout.close()


	fig, ax = plt.subplots()

	ax.plot(myList,k1,'-')
	ax.plot(l1,l,'-')
	ax.set_ylim(bottom=0)
	arrx=np.array(l1)
	yield_st=np.array(k1)
	arr2=np.array(l)
	idx = np.argwhere(np.diff(np.sign(arr2 - yield_st))).flatten()


	plt.plot(arrx[idx],yield_st[idx],'ro')
	plt.xlabel('Strain')
	plt.ylabel('0.2% yield strength (MPa)')
	print('yield strength calculated is'+ str(yield_st[idx])+ 'MPa')
	plt.savefig('{}/stress-strain.png'.format(DPADict['CALC_DIR']))    
    
