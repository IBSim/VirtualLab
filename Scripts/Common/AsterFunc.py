import os
import numpy as np
import sys
from bisect import bisect_left as bl

import aster
from Utilitai import partition
from code_aster.Cata.Syntax import *
from code_aster.Cata.DataStructure import *
from code_aster.Cata.Commons import *
from code_aster.Cata.Commands import *
from Noyau.N__F import _F

def AdaptThermal(ResName,Tsteps,Load,Material,Model,Theta,Solver,**kwargs):
	MaxIter = kwargs.get('MaxIter', 10)

	if 'Storing' in kwargs.keys(): _SaveSteps = _F(LIST_INST=kwargs['Storing'])
	else: _SaveSteps = _F()

	# timearr = np.array(Tsteps.Valeurs())
	tsteps = Tsteps.Valeurs()
	StartTime = (ResName.LIST_VARI_ACCES()['INST'])[-1]
	# StartIx = np.argmin(np.abs(np.array(timearr) - StartTime))

	pos = bl(tsteps, StartTime)
	if abs(tsteps[pos-1]-StartTime) < abs(tsteps[pos]-StartTime): StartIx = pos -1
	else: StartIx = pos


	EndIndex = kwargs.get('EndIndex', None)
	EndTime = kwargs.get('EndTime', None)
	if EndIndex : tsteps = tsteps[StartIx:EndIndex+1]
	elif EndTime: tsteps = tsteps[StartIx:np.argmin(np.abs(np.array(tsteps)-EndTime))+1]
	else: tsteps = tsteps[StartIx:]

	_adapt = DEFI_LIST_REEL(VALE=tsteps)
	### Set err=1 here to enter while loop
	count, err = 0, 1
	while err:
		try:
			THER_NON_LINE(reuse=ResName,
						ARCHIVAGE=_SaveSteps,
						CHAM_MATER=Material,
						ETAT_INIT=_F(EVOL_THER=ResName),
						EXCIT=Load,
						INCREMENT=_F(LIST_INST=_adapt),
						MODELE=Model,
						PARM_THETA=Theta,
						SOLVEUR=_F(METHODE=Solver),
						CONVERGENCE=_F(ITER_GLOB_MAXI=MaxIter,),
						RESULTAT=ResName)
			err=0

		except aster.NonConvergenceError as message:
			err=1
			count += 1
			ChangeIx = StartIx + message.vali[0]
			ProbStep = tsteps[ChangeIx-1:ChangeIx+1]
			h=0.5
			NewStep = (1-h)*ProbStep[0] + h*ProbStep[1]
			# TstepSize = timearr[ChangeIx] - timearr[ChangeIx - 1]
#			print ('The problem timestep is number {}. Previously it was {}, but has been updated to {}'.format(str(ChangeIx),TstepSize,TstepSize/2))

			# timearr = np.insert(timearr,ChangeIx,timearr[ChangeIx-1] + TstepSize/2)
			tsteps.insert(ChangeIx,NewStep)
			DETRUIRE(CONCEPT=_F(NOM=(_adapt)))
			_adapt = DEFI_LIST_REEL(VALE=tsteps)

	DETRUIRE(CONCEPT=_F(NOM=(_adapt)))
	# print('{} timesteps have been added to the list of timesteps'.format(count))
	return tsteps

def MaterialProps(Mat_dir,Materials):
	## Removes any duplicate materials to save time
	if type(Materials) == str: Materials = [Materials]
	Materials = set(Materials)
	PropDict = {}
	for Mat in Materials:
		PropPath = '{}/{}'.format(Mat_dir,Mat)
		PropDict[Mat] = {}
		for (rootdir, dirnames, filenames) in os.walk(PropPath):
			for files in filenames:
				name, ext = os.path.splitext(files)
				fname = "{}/{}".format(PropPath,files)
				data = np.fromfile(fname,dtype=float,count=-1,sep=" ")
				if name == 'Rho': RhoOrig = data
				elif name == 'Cp': CpOrig = data
				# Create a flat line if number of values is 1 or 2 (not enough data to be NonLinear)
				if len(data) in (1,2):
					data = np.array([0,data[-1],100,data[-1]])
				PropDict[Mat][name] = data

		### Uses Rho & CP to make distribution for RhoCP
		if len(RhoOrig) <= 2 and len(CpOrig) <=2:
			RhoCpval = RhoOrig[-1]*CpOrig[-1]
			RhoCp = np.array([0, RhoCpval, 100, RhoCpval])
		elif len(RhoOrig) > 2 and len(CpOrig) <= 2:
			RhoCp = RhoOrig
			RhoCp[1::2] = RhoOrig[1::2]*CpOrig[-1]
		elif len(RhoOrig) <= 2 and len(CpOrig) > 2:
			RhoCp = CpOrig
			RhoCp[1::2] = CpOrig[1::2]*RhoOrig[-1]
		else :
			minval, maxval = min(RhoOrig[0], CpOrig[0]), max(RhoOrig[-2], CpOrig[-2])
			datax = np.linspace(minval,maxval,11)
			datay = np.interp(datax,RhoOrig[0::2],RhoOrig[1::2])*np.interp(datax,CpOrig[0::2],CpOrig[1::2])
			RhoCp = [None]*(len(datax)*2)
			RhoCp[::2], RhoCp[1::2] = datax, datay
		PropDict[Mat]['RhoCp'] = RhoCp

	return PropDict

def BCinfo(meshname,**kwargs):
	mesht = partition.MAIL_PY()
	mesht.FromAster(meshname)

	if 'Group' in kwargs.keys():
		ElemID = mesht.gma.get(kwargs['Group'])
	elif 'Elements' in kwargs.keys():
		ElemID = kwargs['Elements']
	else:
		print('No group or element list have been provided')
		return None, None, None

	NNodes = len(mesht.cn)
	Nodal = np.zeros((NNodes),dtype = float)
	Nodelist = []
	for element in ElemID:
		Nodes = mesht.co[element]
		Nodelist += list(Nodes)

		num = len(Nodes)
		if  num == 3:
			coor1 = np.array(mesht.cn[Nodes[0]])
			coor2 = np.array(mesht.cn[Nodes[1]])
			coor3 = np.array(mesht.cn[Nodes[2]])
			measure = 0.5*np.linalg.norm(np.cross(coor2-coor1,coor3-coor1))

		elif num == 4:
			coor1 = np.array(mesht.cn[Nodes[0]])
			coor2 = np.array(mesht.cn[Nodes[1]])
			coor3 = np.array(mesht.cn[Nodes[2]])
			coor4 = np.array(mesht.cn[Nodes[3]])
			measure = 1/float(6)*abs(np.dot(np.cross(coor2-coor1,coor3-coor1),coor4 - coor1))


		for node in Nodes:
			Nodal[node] = Nodal[node] + measure/num


	Nodal2 = Nodal[Nodal!=0] ## Remove zero rows
	NodeID = sorted(set(Nodelist))
	return Nodal2, NodeID, ElemID


def EMloading(mesh, EMpath, groups, scaling = 1, Tol = 0.5):
	mesht = partition.MAIL_PY()
	mesht.FromAster(mesh)

	EMdat = np.fromfile(EMpath,dtype=float,count=-1,sep=" ")
	NNodes = len(mesht.cn)

	EMdat = EMdat.reshape((NNodes+1,len(EMdat)//(NNodes+1)))

	Temps, JouleHeat = EMdat[0,:], EMdat[1:,:]
	NTemps = len(Temps)

	listest = []
	Load_EM, Elem_EM,  = [], []
	loadlist, meshlist = [], []
	count = 0
	ld = [None]
	for grp in list(groups):
		for i,element in enumerate(mesht.gma.get(grp)):
			Nodes = mesht.co[element]
			load = 0
			for node in Nodes:
				load = load + JouleHeat[node,:]/4

			listest.append(load)
			loadlist.append(scaling*load)

			if max(load) < Tol:
				continue

			load = scaling*load

			if NTemps == 1:
				### Create 2 points for a flat constant line
				vals = [Temps[0],load[0],Temps[0]+1,load[0]]
			else:
				vals = [None]*(2*NTemps)
				vals[::2] = Temps
				vals[1::2] = load

	#		ld[count] = DEFI_CONSTANTE(VALE=load[0])
			ld[count] = DEFI_FONCTION(NOM_PARA='TEMP',
					       PROL_DROITE='CONSTANT',
					       PROL_GAUCHE='CONSTANT',
					       VALE=vals)

			meshlist.append(_F(GROUP_MA=(grp), NOM='M{}'.format(element), NUME_FIN=i+1, NUME_INIT=i+1))
			Load_EM.append((_F(GROUP_MA=('M{}'.format(element), ),SOUR=ld[count])))
			Elem_EM.append(element)

			ld.append(None)
			count = count+1

	### Create all the groups at the end to speed up the process
	DEFI_GROUP(reuse=mesh,
		   MAILLAGE=mesh,
		   CREA_GROUP_MA=meshlist)

	return Load_EM, Elem_EM, EMdat


def Timesteps(dt, start=0):
	timelist, savelist = [], []
	for i, tup in enumerate(dt):
		if len(tup) == 3: dt, Nstep, save = tup
		else :dt, Nstep, save = tup[0],tup[1], 1

		fintime = start + dt*Nstep
		timesteps = np.round(np.linspace(start,fintime,Nstep+1),14).tolist()

		if i == 0:
			timelist = timesteps
			savelist = timesteps[::save]
		else:
			timelist += timesteps[1:]
			savelist += timesteps[save::save]
		start = fintime

	return timelist, savelist
