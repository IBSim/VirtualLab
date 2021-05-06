import h5py
import numpy as np
import os
import sys
from VLFunctions import MeshInfo, MaterialProperty
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import scipy.ndimage
from importlib import import_module

def Single(Info, StudyDict): 
	Parameters = StudyDict["Parameters"]
	ResFile = '{}/Thermal.rmed'.format(StudyDict['ASTER']) 
#============================================================================
	# open result file using h5py
	g = h5py.File(ResFile, 'r')
	gRes = g['/CHA/Temperature']
	steps = gRes.keys() # temp values only in last step
	time = [gRes[step].attrs['PDT'] for step in steps]



	# If mesh info file exists import it
	if os.path.isfile("{}/{}.py".format(Info.MESH_DIR,Parameters.Mesh)): 
		sys.path.insert(0,Info.MESH_DIR)

#============================================================================
	# Get mesh information from the results file
		meshdata = MeshInfo(ResFile)     
		SurfaceNormal = [['TileFront', 'NX'], ['TileBack', 'NX'], ['TileSideA', 'NY'], ['TileSideB', 'NY'], ['TileTop', 'NZ'], ['BlockFront', 'NX'], ['BlockBack', 'NX'], ['BlockSideA', 'NY'], ['BlockSideB', 'NY'],['BlockBottom', 'NZ'], ['BlockTop', 'NZ']]         
		NodesID = []
		NodeNumber = []  #number of nodes and elements in each surface
		ElementNumber = []
		ElementNodes = []  # element connectivity
		alpha = [] # position vector
		CornerCoord = [] # corner coordinates of tile/block/..surfaces
		cSurfaceNames = [] # correctly defined surfaces from Parameters.ThermoCouple (in input file)
		for thermo in Parameters.ThermoCouple:
			surfaceName = thermo[0] + thermo[1]
			dummy_iterator = 0

			for iterator in SurfaceNormal:
				if iterator[0] == surfaceName:
					cSurfaceNames.append([surfaceName, thermo[2], thermo[3]])
					ObjectSurface = meshdata.GroupInfo(surfaceName)
					NodesID.append(ObjectSurface.Nodes) # Node id lists
					NodeNumber.append(ObjectSurface.NbNodes)
					ElementNumber.append(ObjectSurface.NbElements)
					ElementNodes.append(ObjectSurface.Connect)
					dummy_iterator = 1
					if iterator[1]=='NX':
						alpha.append([0, thermo[2], thermo[3]])
					elif iterator[1]=='NY':
						alpha.append([thermo[2], 0, thermo[3]])
					else:
						alpha.append([thermo[2], thermo[3], 0])
			if dummy_iterator == 0:
				print("warning: there is no surface with the name %s in Sample" %surfaceName)

		SurfaceN = len(alpha) # number of surfaces where we can mount thermocouples
#============================================================================
	#open an empty file to write average temperatures of the selected areas in which thermocouple are placed 
		if Parameters.TemperatureOut == True:
			FileTemp = open('ThermocoupleTemp.txt', 'w')
			FileTemp.write('Time ')
			for SearchRadius in Parameters.Rvalues:
				for name in cSurfaceNames:
					output = name[0] + '_' + str(round(name[1], 2)) + '_' + str(round(name[2], 2)) + '_' + str(SearchRadius) + ' '
					FileTemp.write(output)
			FileTemp.write('\n')
#============================================================================		
		for SurfaceID in range(SurfaceN):
			temp = meshdata.GetNodeXYZ(NodesID[SurfaceID]) # temp: temporary variable
			minX = min(x for (x, y, z) in temp)
			minY = min(y for (x, y, z) in temp)
			minZ = min(z for (x, y, z) in temp)
			maxX = max(x for (x, y, z) in temp)
			maxY = max(y for (x, y, z) in temp)
			maxZ = max(z for (x, y, z) in temp)

			CornerCoord.append([minX, minY, minZ, maxX, maxY, maxZ])

#============================================================================ 
		# Create list for storing nodal IDs found in radii of search
		SearchNodeID = []
		SearchNbNodesN = []
		DummyListLocal = [] # temporary list 

		for SurfaceID in range(SurfaceN):
			SearchX = CornerCoord[SurfaceID][0] + alpha[SurfaceID][0]*(CornerCoord[SurfaceID][3] - CornerCoord[SurfaceID][0])
			SearchY = CornerCoord[SurfaceID][1] + alpha[SurfaceID][1]*(CornerCoord[SurfaceID][4] - CornerCoord[SurfaceID][1])
			SearchZ = CornerCoord[SurfaceID][2] + alpha[SurfaceID][2]*(CornerCoord[SurfaceID][5] - CornerCoord[SurfaceID][2])
			for SearchRadius in Parameters.Rvalues:
				SearchNbNodes = 0

				for node in NodesID[SurfaceID]:
					TempNodesID = NodesID[SurfaceID]
					NodeXYZ = meshdata.GetNodeXYZ(node)
					Distance = np.sqrt((NodeXYZ[0]-SearchX)**2 + (NodeXYZ[1]-SearchY)**2 + (NodeXYZ[2]-SearchZ)**2)

					if Distance <= SearchRadius:
						DummyListLocal.append(node) # nodes within radius of search
						SearchNbNodes += 1
				SearchNbNodesN.append([SurfaceID, SearchRadius, SearchNbNodes])
				SearchNodeID += DummyListLocal
				DummyListLocal.clear()

				if SearchNbNodes == 0:
					print('warning!!..no nodes were found in location %f and %f on surface %s within search radius %f' %( cSurfaceNames[SurfaceID][1], cSurfaceNames[SurfaceID][2], cSurfaceNames[SurfaceID][0], SearchRadius) )
					print('..either increase radius of search or use smaller mesh density!')
#============================================================================ 
		# Read nodal temperature from .h5py file and write in ThermocoupleTemp.txt file
		cstep = []
		ctime = []

		if Parameters.CaptureTime == 'all': # finding the list of steps for all time increments
			iterator = len(steps)
			for time1, step1 in zip(time, steps):
				ctime.append(time1)
				cstep.append(step1)
		else: # finding a single step for a specific increment assigned in input file
			iterator = 1
			for time1, step1 in zip(time, steps):
				if time1 == Parameters.CaptureTime:
					cstep.append(step1)
					ctime.append(time1)
	
		averageTemp = [] # store average nodal temp over thermocouples
		for g in range(iterator): # time loop using the list of step or single step

			TemperatureNodes = gRes['{}/NOE/MED_NO_PROFILE_INTERNAL/CO'.format(cstep[g])][:]
			
			k_old = 0
			if Parameters.TemperatureOut == True:
				output = str(ctime[g]) + ' '
				FileTemp.write(output)

			for m in range(len(SearchNbNodesN)):
				TemperatureAve, TemperatureNode, TemperatureSum = 0.0, 0.0, 0.0
				for k in range(SearchNbNodesN[m][2]): 
					TemperatureSum += TemperatureNodes[SearchNodeID[k+k_old]]

				k_old += SearchNbNodesN[m][2] 

				if SearchNbNodesN[m][2] >= 1:
					TemperatureAve = TemperatureSum/float (SearchNbNodesN[m][2])
					if Parameters.TemperatureOut == True:
						output = str (format(TemperatureAve, '.2f')) + ' '
						FileTemp.write(output)
					averageTemp.append([cSurfaceNames[m][0], TemperatureAve])
				else:
					if Parameters.TemperatureOut == True:
						FileTemp.write('NaN ')
					averageTemp.append([cSurfaceNames[m][0], 'NaN'])
			if Parameters.TemperatureOut == True:
				FileTemp.write('\n')
#============================================================================ 
		# Temperature Plots

		if Parameters.TemperaturePlot == True: 
			aveTemp = [] # average temperature over time
			fig = plt.figure(figsize = (14,5))
			plt.xlabel('Time (second)',fontsize = 16)
			plt.ylabel('Temperature (Celcius)',fontsize = 16)
					
			for i in range(len(cSurfaceNames)):
				[aveTemp.append(a[1]) for a in averageTemp if cSurfaceNames[i][0] == a[0]]		
				if aveTemp[0]!= 'NaN':
					label1 = 'Avg. Temperature over Thermocouple on '+ cSurfaceNames[i][0]
					plt.plot(ctime, aveTemp, label = label1)
				aveTemp.clear()
			plt.legend(loc='upper left')
			imageName = "{}/AvgTemperatureThermocouples.png"
			plt.savefig(imageName.format(StudyDict['POSTASTER']), bbox_inches='tight')
			print("Created plot " +imageName +"\n")
			plt.close()
				

	else :
		print('mesh file is not found!!')#pass


	if Parameters.TemperatureOut == True:
		FileTemp.close()



