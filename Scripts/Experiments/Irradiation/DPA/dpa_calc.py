import shutil
import os
import sys
sys.dont_write_bytecode=True
import numpy as np
import h5py
import matplotlib.pyplot as plt
import time
from bisect import bisect_left as bl
import shutil
from subprocess import Popen




def dpa_calculation(VL,DPADict):
	Parameters = DPADict["Parameters"]
	
	# Calculation of dpa for copper interlayer

	h2=(getattr(Parameters,'width_mesh',None)) # Mesh size along the width used from neutronics simulation
	w2=(getattr(Parameters,'height_mesh',None)) # Mesh size along the height used from neutronics simulation
	t2=(getattr(Parameters,'thic_mesh',None)) # Mesh size along the thickness used from neutronics simulation
	hwt=h2*w2*t2 # Total Mesh elements
        
	# Damage energy on finite element mesh
	f = open('{}/damagecu'.format(DPADict['CALC_DIR']),'r')
	lines=f.readlines()
	noden = [(line.strip().split())for line in lines]
	e=len(noden)

	TT=[]
	TT2=[]
	TT1=[]
	TT3=[]
	for j in range(0,e):
            TT.append(str(int(noden[j][0])+int(1))) 
            TT1.append(float(noden[j][1])) 
            TT3.append((float(noden[j][2]))) 
            
	max1=max(TT3)
      	
	# Create cluster for damage energy
	NbClusters=getattr(Parameters,'Cluster_cu',None)

	JH_Vol1 = np.array(TT1)*max1
	sum1=np.sum(JH_Vol1)
	print(sum1)
	JH_Vol=JH_Vol1.astype(np.float64)
	from sklearn.cluster import KMeans
	elements1 = np.array(TT)
        
	Elements=elements1.astype(np.int64)		
	X = JH_Vol.reshape(-1,1)

	X_sc = (X - X.min())/(X.max()-X.min())
	kmeans = KMeans(n_clusters=NbClusters).fit(X_sc)

		# Goodness of Fit Value is a metric of how good the clustering is
	SDAM = ((X_sc - X_sc.mean())**2).sum() # Squared deviation for mean array
	SDCM = kmeans.inertia_ # Squared deviation class mean
	GFV = (SDAM-SDCM)/SDAM # Goodness of fit value

	EM_Groups = [Elements[kmeans.labels_==i] for i in range(NbClusters)]
	EM_Loads = kmeans.cluster_centers_*(X.max()-X.min()) + X.min()


	sorted_y, sorted_x = zip(*sorted(zip(EM_Loads, EM_Groups)))
	myFileNamecu ='damagecu'+ str(int(getattr(Parameters,'dpa',None)))+'_dpa'
	myFileNamecu1 ='damagecu'+ str(int(getattr(Parameters,'dpa',None)))+'_dpa_groups'
	path6 = os.path.join('{}'+'/'+ myFileNamecu)
	np.savetxt(path6.format(DPADict['CALC_DIR']), sorted_y, delimiter = '\t') 
	dpacu='dpacu'+ str(int(getattr(Parameters,'dpa',None)))
	path6=os.path.join('{}'+'/'+ dpacu)
	f6=open(path6.format(DPADict['CALC_DIR']),"w")

	# Calculation of dpa for copper interlayer
	for i in range(len(sorted_x)):
	    damagecu='damagecu'+ str(int(getattr(Parameters,'dpa',None)))+'_dpa'+str(i)
	    path6=os.path.join('{}'+'/'+ damagecu)
	    f=open(path6.format(DPADict['CALC_DIR']),"w")
            
	    ff=sorted_y[i][0]*len(sorted_x[i])
	    h=getattr(Parameters,'Warmour_height_lower',None)
	    h1=getattr(Parameters,'Warmour_height_upper',None)
	    w=getattr(Parameters,'Warmour_width',None)
	    th=getattr(Parameters,'Warmour_thickness',None)
	    vol1=(h+h1)*w*th/hwt
	    vol=max1*vol1*len(sorted_x[i])
  	    
	    displacements_per_source_neutron = ff/ (2*20)

	    displacements_per_source_neutron_with_recombination = displacements_per_source_neutron*0.8
   
	    fusion_power = getattr(Parameters,'fusion_power',None)  # units Watts
	    energy_per_fusion_reaction = 14e6  # units eV
	    eV_to_Joules = 1.60218e-19  # multiplication factor to convert eV to Joules
	    number_of_neutrons_per_second = fusion_power / (energy_per_fusion_reaction * eV_to_Joules)

	    number_of_neutrons_per_year = number_of_neutrons_per_second * 60 * 60 *24*(int(getattr(Parameters,'days',None)))
   
	    displacements_for_all_atoms = number_of_neutrons_per_year * displacements_per_source_neutron_with_recombination
	    copper_atomic_mass_in_g = 64*1.66054E-24  # molar mass multiplier by the atomic mass unit (u)
	    number_of_copper_atoms =  vol* 8.9 / (copper_atomic_mass_in_g)
	    DPA = displacements_for_all_atoms / number_of_copper_atoms
	    path7=os.path.join('{}'+'/'+ dpacu)
	    f6=open(path7.format(DPADict['CALC_DIR']),"a")
	    f6.write(str(float(DPA))+"\n") 
   
    
	    for j in range(len(sorted_x[i])):
	        f.write(str(float(ff))+' '+str(float(DPA))+' '+str(int(sorted_x[i][j]))+' '+str(i)+"\n") 


	
	# Calculation of dpa for Tungsten armour
	
	f = open('{}/damagetu'.format(DPADict['CALC_DIR']),'r')
	lines=f.readlines()
	noden = [(line.strip().split())for line in lines]
	e=len(noden)

	TT=[]
	TT2=[]
	TT1=[]
	TT3=[]
	for j in range(0,e):
            TT.append(str(int(noden[j][0])+int(1))) 
   
            TT1.append(float(noden[j][1])) 
            TT3.append((float(noden[j][2]))) 
	max1=max(TT3)
          
   
	NbClusters=getattr(Parameters,'Cluster_tu',None)

	JH_Vol1 = np.array(TT1)*max1
	sum2=np.sum(JH_Vol1)
	print(sum2)
	JH_Vol=JH_Vol1.astype(np.float64)
	from sklearn.cluster import KMeans
	elements1 = np.array(TT)

	Elements=elements1.astype(np.int64)		
	X = JH_Vol.reshape(-1,1)

	X_sc = (X - X.min())/(X.max()-X.min())
	kmeans = KMeans(n_clusters=NbClusters).fit(X_sc)

		# Goodness of Fit Value is a metric of how good the clustering is
	SDAM = ((X_sc - X_sc.mean())**2).sum() # Squared deviation for mean array
	SDCM = kmeans.inertia_ # Squared deviation class mean
	GFV = (SDAM-SDCM)/SDAM # Goodness of fit value

	EM_Groups = [Elements[kmeans.labels_==i] for i in range(NbClusters)]
	EM_Loads = kmeans.cluster_centers_*(X.max()-X.min()) + X.min()


	sorted_y, sorted_x = zip(*sorted(zip(EM_Loads, EM_Groups)))
	s=0
	myFileNamecu ='damagetu'+ str(int(getattr(Parameters,'dpa',None)))+'_dpa'
	myFileNamecu1 ='damagetu'+ str(int(getattr(Parameters,'dpa',None)))+'_dpa_groups'
	path6 = os.path.join('{}'+'/'+ myFileNamecu)
	np.savetxt(path6.format(DPADict['CALC_DIR']), sorted_y, delimiter = '\t') 
	dpatu='dpatu'+ str(int(getattr(Parameters,'dpa',None)))
	path6=os.path.join('{}'+'/'+ dpatu)
	f6=open(path6.format(DPADict['CALC_DIR']),"w")
	for i in range(len(sorted_x)):
	    damagetu='damagetu'+ str(int(getattr(Parameters,'dpa',None)))+'_dpa'+str(i)
	    path6=os.path.join('{}'+'/'+ damagetu)
	    f=open(path6.format(DPADict['CALC_DIR']),"w")
	    ff=sorted_y[i][0]*len(sorted_x[i])
	    s=s+len(sorted_x[i])
	    h=getattr(Parameters,'Warmour_height_lower',None)
	    h1=getattr(Parameters,'Warmour_height_upper',None)
	    w=getattr(Parameters,'Warmour_width',None)
	    th=getattr(Parameters,'Warmour_thickness',None)
	    vol1=(h+h1)*w*th/hwt
	    vol=max1*vol1*len(sorted_x[i])
  
	    displacements_per_source_neutron = ff/ (2*90)

	    displacements_per_source_neutron_with_recombination = displacements_per_source_neutron*0.98
   
	    fusion_power = getattr(Parameters,'fusion_power',None)  # units Watts
	    energy_per_fusion_reaction = 14e6  # units eV
	    eV_to_Joules = 1.60218e-19  # multiplication factor to convert eV to Joules
	    number_of_neutrons_per_second = fusion_power / (energy_per_fusion_reaction * eV_to_Joules)

	    number_of_neutrons_per_year = number_of_neutrons_per_second * 60 * 60 *24*(int(getattr(Parameters,'days',None)))
   
	    displacements_for_all_atoms = number_of_neutrons_per_year * displacements_per_source_neutron_with_recombination
	    tungsten_atomic_mass_in_g = 183*1.66054E-24  # molar mass multiplier by the atomic mass unit (u)
	    number_of_tungsten_atoms =  vol* 19.6 / (tungsten_atomic_mass_in_g)
	    DPA = displacements_for_all_atoms / number_of_tungsten_atoms
	    path7=os.path.join('{}'+'/'+ dpatu)
	    f5=open(path7.format(DPADict['CALC_DIR']),"a")
	    f5.write(str(float(DPA))+"\n") 
   
    
	    for j in range(len(sorted_x[i])):
	        f.write(str(float(ff))+' '+str(float(DPA))+' '+str(int(sorted_x[i][j]))+' '+str(i)+"\n") 

    
	# Calculation of dpa for CuCrZr coolant pipe
	
	f = open('{}/damagecur'.format(DPADict['CALC_DIR']),'r')
	lines=f.readlines()
	noden = [(line.strip().split())for line in lines]
	e=len(noden)

	TT=[]
	TT2=[]
	TT1=[]
	TT3=[]
	for j in range(0,e):
            TT.append(str(int(noden[j][0])+int(1))) 
            TT1.append(float(noden[j][1])) 
            TT3.append((float(noden[j][2])))  
            
	max1=max(TT3)
          
	NbClusters=getattr(Parameters,'Cluster_cucrzr',None)

	JH_Vol1 = np.array(TT1)*max1
	
	sum3=np.sum(JH_Vol1)
	print(sum3)
	JH_Vol=JH_Vol1.astype(np.float64)
	from sklearn.cluster import KMeans
	elements1 = np.array(TT)

	Elements=elements1.astype(np.int64)		
	X = JH_Vol.reshape(-1,1)

	X_sc = (X - X.min())/(X.max()-X.min())
	kmeans = KMeans(n_clusters=NbClusters).fit(X_sc)

		# Goodness of Fit Value is a metric of how good the clustering is
	SDAM = ((X_sc - X_sc.mean())**2).sum() # Squared deviation for mean array
	SDCM = kmeans.inertia_ # Squared deviation class mean
	GFV = (SDAM-SDCM)/SDAM # Goodness of fit value

	EM_Groups = [Elements[kmeans.labels_==i] for i in range(NbClusters)]
	EM_Loads = kmeans.cluster_centers_*(X.max()-X.min()) + X.min()


	sorted_y, sorted_x = zip(*sorted(zip(EM_Loads, EM_Groups)))
	myFileNamecu ='damagecucr'+ str(int(getattr(Parameters,'dpa',None)))+'_dpa'
	myFileNamecu1 ='damagecucr'+ str(int(getattr(Parameters,'dpa',None)))+'_dpa_groups'
	path6 = os.path.join('{}'+'/'+ myFileNamecu)
	np.savetxt(path6.format(DPADict['CALC_DIR']), sorted_y, delimiter = '\t') 
	dpacucr='dpacucr'+ str(int(getattr(Parameters,'dpa',None)))
	path6=os.path.join('{}'+'/'+ dpacucr)
	f6=open(path6.format(DPADict['CALC_DIR']),"w")
	for i in range(len(sorted_x)):
	    damagecucr='damagecucr'+ str(int(getattr(Parameters,'dpa',None)))+'_dpa'+str(i)
	    path6=os.path.join('{}'+'/'+ damagecucr)
	    f=open(path6.format(DPADict['CALC_DIR']),"w")
	    ff=sorted_y[i][0]*len(sorted_x[i])
	    h=getattr(Parameters,'Warmour_height_lower',None)
	    h1=getattr(Parameters,'Warmour_height_upper',None)
	    w=getattr(Parameters,'Warmour_width',None)
	    th=getattr(Parameters,'Warmour_thickness',None)
	    vol1=(h+h1)*w*th/hwt
	    vol=max1*vol1*len(sorted_x[i])
	    displacements_per_source_neutron = ff/ (2*40)

	    displacements_per_source_neutron_with_recombination = displacements_per_source_neutron*0.8
   
	    fusion_power = getattr(Parameters,'fusion_power',None) # units Watts
	    energy_per_fusion_reaction = 14e6  # units eV
	    eV_to_Joules = 1.60218e-19  # multiplication factor to convert eV to Joules
	    number_of_neutrons_per_second = fusion_power / (energy_per_fusion_reaction * eV_to_Joules)

	    number_of_neutrons_per_year = number_of_neutrons_per_second * 60 * 60 *24*(int(getattr(Parameters,'days',None)))
   
	    displacements_for_all_atoms = number_of_neutrons_per_year * displacements_per_source_neutron_with_recombination
	    cucrzr_atomic_mass_in_g = 64*1.66054E-24  # molar mass multiplier by the atomic mass unit (u)
	    number_of_cucrzr_atoms =  vol* 8.9 / (cucrzr_atomic_mass_in_g)
	    DPA = displacements_for_all_atoms / number_of_cucrzr_atoms
	    path7=os.path.join('{}'+'/'+ dpacucr)
	    f5=open(path7.format(DPADict['CALC_DIR']),"a")
	    f5.write(str(float(DPA))+"\n") 
   
    
	    for j in range(len(sorted_x[i])):
	        f.write(str(float(ff))+' '+str(float(DPA))+' '+str(int(sorted_x[i][j]))+' '+str(i)+"\n")  
