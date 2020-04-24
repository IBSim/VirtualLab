import h5py
import sys
import numpy as np
sys.dont_write_bytecode=True

def MaterialProperty(matarr,Temperature):
	if len(matarr) in (1,2): return matarr[-1]
	else: return np.interp(Temperature, matarr[::2], matarr[1::2])

class MeshInfo():
	def __init__(self, meshfile, meshname = None):
		self.g = h5py.File(meshfile, 'r')
		# Fin the name of the mesh(es) in the file
		names = self.g['ENS_MAA'].keys()
		
		# If only one mesh in file then set meshname to this
		if len(names) == 1:
			meshname = list(names)[0]
		# If meshname provided check it is in the file, if not error
		elif meshname and str(meshname) not in names:
			err = 'meshname provided not in file'
		# If multiple meshes in the file and no meshname provided, call error
		else :
			err = 'Multiple meshes in file and no meshname given'

		self.MeshName = meshname

		# CnctPath is the path to nodal and element data
		self.CnctPath = "ENS_MAA/{}/-0000000000000000001-0000000000000000001".format(self.MeshName)
		self.NbNodes = self.g["{}/NOE/COO".format(self.CnctPath)].attrs['NBR']

		self.NbElements, self.NbVolumes, self.NbSurfaces, self.NbEdges = 0, 0, 0, 0
		for ElType in self.g["{}/MAI".format(self.CnctPath)].keys():
			value = self.g["{}/MAI/{}/NUM".format(self.CnctPath, ElType)].attrs['NBR']
			if ElType == 'TE4':
				self.NbVolumes = value
			elif ElType == 'TR3':
				self.NbSurfaces = value
			elif ElType == 'SE2':
				self.NbEdges = value
			self.NbElements += value

	def __GroupSort__(self):
		grpInfo = {}
		for ElType in self.g["{}/MAI".format(self.CnctPath)].keys():
			for val in np.unique(self.g["{}/MAI/{}/FAM".format(self.CnctPath, ElType)][:]):
				grpInfo[val] = ElType

		Groups = {}
		# Element groups
		ElGrpPath = "FAS/{}/ELEME".format(self.MeshName)
		if ElGrpPath in self.g:
			for Grp in self.g[ElGrpPath].keys():
				# Get unique number associated to elements for grouping
				num = self.g["{}/{}".format(ElGrpPath, Grp)].attrs['NUM']

				grptype = grpInfo[num]
				if grptype not in Groups.keys():
					Groups[grptype] = {}

				# Find the group(s) associated with this unique number
				for uniname in self.g["{}/{}/GRO/NOM".format(ElGrpPath, Grp)][:]:
					# Convert name from unicode chars to ascii string
					charlist = list(map(chr,uniname))
					asciiname = ''.join(charlist).rstrip(charlist[-1])

					if asciiname not in Groups[grptype].keys():
						Groups[grptype][asciiname] = []

					Groups[grptype][asciiname].append(num)

		# Node groups
		Groups['NODE'] = {}
		NdGrpPath = "FAS/{}/NOEUD".format(self.MeshName)
		if NdGrpPath in self.g:
			for Grp in self.g[NdGrpPath].keys():
				# Get unique number associated to elements for grouping
				num = self.g["{}/{}".format(NdGrpPath, Grp)].attrs['NUM']
				# Find the group(s) associated with this unique number
				for uniname in self.g["{}/{}/GRO/NOM".format(NdGrpPath, Grp)][:]:
					# Convert name from unicode chars to ascii string
					charlist = list(map(chr,uniname))
					asciiname = ''.join(charlist).rstrip(charlist[-1])

					if asciiname not in Groups['NODE'].keys():
						Groups['NODE'][asciiname] = []

					Groups['NODE'][asciiname].append(num)

		self.__GroupInfo__ = Groups

	def GroupTypes(self):
		# This function return a dict of what groups are in the mesh by their type
		if not hasattr(self,'__GroupInfo__'):
			self.__GroupSort__()
		GrpDict = dict.fromkeys(self.__GroupInfo__)
		for key, item in self.__GroupInfo__.items():
			GrpDict[key] = list(item.keys())
		return GrpDict

	def GroupNames(self):
		# This function returns a list of the group names in the mesh
		if not hasattr(self,'__GroupInfo__'):
			self.__GroupSort__()

		return sum([list(item.keys()) for item in self.__GroupInfo__.values()], [])


	def GroupInfo(self, name, GroupType = None):
		if not hasattr(self,'__GroupInfo__'):
			self.__GroupSort__()
				
		grptype = [x for x in self.__GroupInfo__.keys() if name in self.__GroupInfo__[x].keys()]

		if len(grptype) == 0:
			err = 'Name not in mesh'
		elif len(grptype) == 1:
			grptype = grptype[0]
		elif len(grptype) > 1 and not GroupType:
			err = 'Multiple groups with same name and no type provided' 
		elif len(grptype) > 1 and GroupType not in grptype:
			err = 'This is not one of the types of groups.' 
		else :
			grptype = GroupType


		# Create empty class to assign information to
		class Groupdata:
			pass
		Groupdata.Type = grptype

		# Element group
		if grptype in ('NODE'):
			path = "{}/NOE".format(self.CnctPath)
			groupbool = np.in1d(self.g["{}/FAM".format(path)][:], self.__GroupInfo__[grptype][name])

			# Find indexes where groupbool is true then add 1 as nodes start from1 not 0 (only works if nodes are in order)
			Groupdata.Nodes = np.where(groupbool)[0] + 1
			Groupdata.NbNodes = len(Groupdata.Nodes)
			# not compatible with rmed files
#			Groupdata.Nodes = self.g["{}/NUM".format(path)][:][groupbool]

		# Element group
		else :
			path = "{}/MAI/{}".format(self.CnctPath,grptype)
			groupbool = np.in1d(self.g["{}/FAM".format(path)][:], self.__GroupInfo__[grptype][name])

			Groupdata.Elements = self.g["{}/NUM".format(path)][:][groupbool]
			Groupdata.NbElements = len(Groupdata.Elements)

			dset = self.g["{}/NOD".format(path)]
			Groupdata.Connect = np.reshape(dset[:], (dset.attrs['NBR'],int(dset.shape[0]/dset.attrs['NBR'])), order='F')[groupbool]
			Groupdata.Nodes = np.unique(Groupdata.Connect)
			Groupdata.NbNodes = len(Groupdata.Nodes)

		return Groupdata


	def GetNodeXYZ(self,nodes):
		nodelist = self.g["{}/NOE/COO".format(self.CnctPath)][:]
		# If individual node is supplied, i.e. of type int of np.int32/64
		if isinstance(nodes, (int, np.integer)):
			xyz = nodelist[np.array([nodes, nodes + self.NbNodes, nodes + 2*self.NbNodes]) -1]
		# If an array or list is passed
		else :
			NdCorrect = np.array(nodes) - 1
			xyz = nodelist[np.concatenate((NdCorrect, NdCorrect + self.NbNodes, NdCorrect + 2*self.NbNodes))]
			xyz = np.reshape(xyz, (len(nodes), 3), order = 'F')

		return xyz



