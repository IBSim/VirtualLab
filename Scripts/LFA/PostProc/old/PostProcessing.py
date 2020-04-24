import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import sys
import os
from textwrap import wrap
sys.dont_write_bytecode=True

def main(TMP_FILE):
	##################################################
	TMP_DIR = os.path.dirname(TMP_FILE)
	f = open(TMP_FILE,'a+')
	Info = f.readlines()
	f.close()
	Dic = {}
	for line in Info:
		data = line.split()
		Dic[data[0][:-1]] = data[1]

	OUTPUT_DIR = Dic['OUTPUT_DIR']
	DATA_DIR = Dic['DATA_DIR']
	IMAGE_DIR = Dic['IMAGE_DIR']
	MATERIAL_DIR = Dic['MATERIAL_DIR']
	LASER_DIR = Dic['LASER_DIR']
	PARAM_DIR = Dic['PARAM_DIR']
	PARAM_MOD = Dic['PARAM_MOD']
	MESH_FILE = Dic['MESH_FILE']

	sys.path.insert(0,PARAM_DIR)
	Param = __import__(PARAM_MOD)
	#################################################


	FluxDist = 'Yes'
	TimeTemp = 'No'
	Rfactor = 'Yes'

	time = np.fromfile(DATA_DIR + '/time.dat',dtype=float,count=-1,sep=" ")

	Coors = np.fromfile(DATA_DIR + '/Coors.dat',dtype=float,count=-1,sep=" ")
	Coors = Coors.reshape((len(Coors)/3,3))

	temp = np.fromfile(DATA_DIR+'/Btemp.dat',dtype=float,count=-1,sep=" ")
	temp = temp.reshape((len(Coors),len(time)),order='F')
	avtemp = np.average(temp,axis=0)

	from PostProc.Interpolate2 import Interp
	from PostProc.LFA_calc_k import KCalc

	l = Param.Height1 + Param.Height2
	vol = (np.pi*Param.Radius**2)*l - (np.pi*Param.VoidRadius**2)*(Param.VoidHeight)
	

	dataRho = np.fromfile(MATERIAL_DIR+'/'+Param.Material+'/Rho.dat',dtype=float,count=-1,sep=" ")
	Rho = Interp(dataRho,Param.ExtTemp)
	dataCp = np.fromfile(MATERIAL_DIR+'/'+Param.Material+'/Cp.dat',dtype=float,count=-1,sep=" ")
	Cp = Interp(dataCp,Param.ExtTemp)
	dataLambda = np.fromfile(MATERIAL_DIR+'/'+Param.Material+'/Lambda.dat',dtype=float,count=-1,sep=" ")
	Lambda = Interp(dataLambda,Param.ExtTemp)

	mass = vol*Rho

	BTemp = np.zeros((len(time),2))
	BTemp[:,0] = time
	BTemp[:,1] = avtemp
#	BTemp[:,0] = time
#	BTemp[:,1] = temp[:,1]
	np.savetxt(DATA_DIR+'/TimeTemp.dat',BTemp,delimiter = '  ')

	HalfTime, EffLambda, EffAlpha = KCalc(BTemp,l,mass,vol,Cp)
	
	NEffLambda = None
	if os.path.isfile(DATA_DIR + '/BtempInd.dat'):
		IndNode = np.fromfile(DATA_DIR+'/BtempInd.dat',dtype=float,count=-1,sep=" ")
		IndNode  = IndNode.reshape((len(IndNode)/2,2))
		junk, NEffLambda, junk = KCalc(IndNode,l,mass,vol,Cp)

	NearestTstep = time[np.argmin(abs(time - float(HalfTime)))]

	f = open(os.path.splitext(MESH_FILE)[0] + '.dat','r+')
	PreProc = f.read()
	f.close()

	f = open(DATA_DIR + '/Aster.dat','r+')
	Aster = f.read()
	f.close()

	g = open(OUTPUT_DIR+'/Information.dat','w+')
	g.write(PreProc)
	g.write(Aster)
	g.write('Half_time: {}\n'.format(HalfTime))
	g.write('Nearest_Tstep: {}\n'.format(NearestTstep))
	g.write('Lambda: {}\n'.format(Lambda))
	g.write('Average_Eff_Lambda: {}\n'.format(EffLambda))
	if NEffLambda:	
		g.write('Node_Eff_Lambda: {}\n'.format(NEffLambda))
#	g.write('Eff_Alpha: {}\n'.format(EffAlpha))
	g.write('Calc_dT: {}\n'.format(avtemp[-1]-Param.InitTemp))
	g.close()

	if FluxDist =='Yes':
		r = 0.007
		fname = DATA_DIR + '/FluxDist.dat'
		file1 = np.fromfile(fname,dtype=float,count=-1,sep=" ")
		file2 = file1.reshape((len(file1)/5,5))

		cmap=cm.coolwarm
		fig = plt.figure(figsize = (14,5))
		ax1 = plt.subplot(121,adjustable = 'box',aspect = 1)
		ax2 = plt.subplot(122,adjustable = 'box',aspect = 1)
		im1 = ax1.scatter(file2[:,0], file2[:,1], c=file2[:,2], cmap=cmap)
		ax1.axis([-r, r, -r, r])
		ax1.axis('off')

		cb = plt.colorbar(im1,ax=ax1)
		im2 = ax2.scatter(file2[:,0], file2[:,1], c=file2[:,4], cmap=cmap)
		ax2.axis([-r, r, -r, r])
		ax2.axis('off')

		plt.colorbar(im2,ax=ax2)
		plt.tight_layout()
		plt.savefig(IMAGE_DIR + '/FluxDist.png',bbox_inches='tight')

	if TimeTemp =='Yes':
		
		fig = plt.figure(figsize = (14,5))
		plt.plot(time,avtemp)
#		plt.title('Average Temperature on bottom surface v time')
		plt.savefig(IMAGE_DIR + '/TimevTemp.png',bbox_inches='tight')

	if Rfactor =='Yes':
		Rfact = Coors[:,2]

		Rvals=[0.1,1]
		fig = plt.figure(figsize = (14,5))
#		fig = plt.figure()
		for c, R in enumerate(Rvals,1):
			Rbool = (Rfact <= R)*1
			Rsum = sum(Rbool)
			AvTemp = np.sum(temp*Rbool[:,np.newaxis],axis=0)/Rsum

			
#			ax = plt.subplot(1,2,c)
#			index = np.where(time < 0.1)[0]
#			print index
			plt.plot(time,AvTemp,label = 'R = {}'.format(R))
#			plt.title("\n".join(wrap("Average temperature for nodes within radius factor R",60)),fontsize = 24)
			plt.xlabel('Time',fontsize = 20)
			plt.ylabel('Temperature',fontsize = 20)
			plt.legend(loc='upper left')
		plt.savefig(IMAGE_DIR +'/Rfactor.png',bbox_inches='tight')



	Laser = np.fromfile(LASER_DIR + '/Laser_{}.dat'.format(Param.Laser),dtype=float,count=-1,sep=" ")
	Laser = Laser.reshape(len(Laser)/2,2)

	fig = plt.figure(figsize = (14,5))
	plt.plot(Laser[:,0],Laser[:,1])
	plt.yticks([0,0.2,0.4,0.6,0.8,1.0], fontsize = 20)
	plt.xticks([0.0002,0.0004,0.0006,0.0008], fontsize = 20)
	plt.xlabel('Time',fontsize=20)
	plt.axis([0, 0.0008, 0, 1.1])
	plt.savefig(IMAGE_DIR + '/LaserPulse.png',bbox_inches='tight')

	return Param.ParaVisFile

