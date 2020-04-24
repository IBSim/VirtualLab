import sys
sys.dont_write_bytecode=True
import numpy as np
from os import walk, path
import inspect

'''
In this script the geometry and mesh we are creating is defined in the function 'Create', with dimensional arguments and mesh arguments passed to it. The 'test' function provides dimensions for when the script is loaded manually in to Salome and not via a parametric study. The error function is imported during the setup of parametric studies to check for any geometrical errors which may arise.
'''


def Create(Parameter,MeshFile):
'''Geometrical and Mesh parameters are stored in the argument Parameter, while the location of where the file will be saved to is stored in Meshfile.'''

	import salome
	salome.salome_init()
	import salome_notebook
	import GEOM
	from salome.geom import geomBuilder
	import math
	import SALOMEDS
	import  SMESH, SALOMEDS
	from salome.smesh import smeshBuilder
	theStudy = salome.myStudy
	notebook = salome_notebook.NoteBook(theStudy)

	print ('\nCreating {}\n'.format(Parameter.MeshName))

	###
	### GEOM component
	###
	

	###
	### SMESH component
	###



def error(Parameters):
''' This function is imported in during the Setup to pick up any errors which will occur for the given geometrical dimension. i.e. impossible dimensions. Set the message variable to the exit error message you would like to see, i.e. 'Cannot have a cube with dimension zero' '''

	message = None
	return message

class Dimensions():
	def __init__(self):
	'''Define geometrical and mesh parameters here which will run when this script is executed in Salome's 'LoadScript' routine. This is used for testing changes.'''
		### Geometric parameters

		### Mesh parameters

		self.MeshName = 'Test'

def Test():
	Parameters = Dimensions()
	Create(Parameters,None)

if __name__ == '__main__':
	Test()
