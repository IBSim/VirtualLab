import os
import sys
from importlib import import_module, reload
import h5py
import numpy as np
from VLFunctions import MeshInfo

def main(Meta):
	sys.path.insert(0,Meta.MESH_DIR)
	Input, Output = [], []
	for SubDir in os.listdir(Meta.STUDY_DIR):
		ResDir = "{}/{}".format(Meta.STUDY_DIR,SubDir)
		if not os.path.isdir(ResDir): continue

		sys.path.insert(0,ResDir)
		Parameters = reload(import_module('Parameters'))
		sys.path.pop(0)

		ERMESres = h5py.File("{}/PreAster/ERMES.rmed".format(ResDir), 'r')
		Watts = ERMESres["EM_Load/Watts"][:]
		JHNode =  ERMESres["EM_Load/JHNode"][:]
		ERMESres.close()

		# Calculate power
		CoilPower = np.sum(Watts)
		# Calculate uniformity
		CoilFace = MeshInfo("{}/{}.med".format(Meta.MESH_DIR,Parameters.Mesh), meshname='Sample').GroupInfo('CoilFace')
		Std = np.std(JHNode[CoilFace.Nodes-1])

		VLMesh = import_module(Parameters.Mesh)

		Input.append(VLMesh.CoilDisplacement + [VLMesh.Rotation])
		Output.append([CoilPower,Std])

	# MLfile = "{}/Data.hdf5".format(Meta.ML_DIR)

	# for SimName, SimDict in Meta.SimData.items():
	# 	Parameters =
	#
	# 	# Get mesh information from {MESHNAME}.py file in MESH_DIR
	# 	sys.path.insert(0,Info.MESH_DIR)
	# 	VLMesh = import_module(Parameters.Mesh)
	#
	# 	CoilTypes = ['Test','HIVE']
	# 	Ix = CoilTypes.index(VLMesh.CoilType)
	# 	InList = [Ix,VLMesh.CoilRotation] + VLMesh.CoilDisplacement
	# 	OutList = [CoilPower,Std]
	# 	Rlist = np.around(InList+OutList,6)
	# 	print(Rlist)
	#
	# 	MLFile = "{}/MLmid.hdf5".format(Info.STUDY_DIR)
	# 	Write = True
	# 	while Write:
	# 		try :
	# 			ML = h5py.File(MLFile, 'a')
	# 			Write = False
	# 		except OSError:
	# 			pass
	# 	# print('writing',Parameters.Name)
	# 	if 'ERMES' not in ML.keys():
	# 		length = Rlist.shape[0]
	# 		ML.create_dataset('ERMES',(length,1),data=Rlist,maxshape=(length,None))
	# 	else:
	# 		ML['ERMES'].resize(ML['ERMES'].shape[1]+1,axis=1)
	# 		ML['ERMES'][:,-1] = Rlist
	# 	ML.close()
