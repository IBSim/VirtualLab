from fpdf import FPDF
import scipy.ndimage
import numpy as np
import os

Factor = float(10)/float(25) ### Font of 25 fits well in a 10mm high box
ShowBox = 0

def add_image(locy,image_path,description):
	height, width, channels = scipy.ndimage.imread(image_path).shape
	AR = float(height)/float(width)

	imwidth = 100
	imheight = imwidth*AR
	locx = 105-(imwidth/2)
	pdf.image(image_path, x=locx, y=locy, w=imwidth)
	pdf.ln(imheight)
	fontsize = 8
	pdf.set_font("Arial",style = 'I', size = fontsize)
	pdf.set_x(locx)
	pdf.cell(100, Factor*fontsize, txt="{}".format(description), border = ShowBox, ln=1, align='C')
	pdf.ln(2)

def paragraph(Text,fontsize,length):
	height = Factor*fontsize
	pdf.set_font("Arial", size=fontsize)

	tmpIntro = Text
	txtlength = pdf.get_string_width(tmpIntro)
	ratio = txtlength/len(tmpIntro)
	while txtlength > length:
		ratio = txtlength/len(tmpIntro)
		chars = int(length/ratio)
		string = tmpIntro[:chars]
		index = (string[::-1]).find(' ')
		end = chars - index
		pdf.cell(length, height, string[:end], ShowBox, 1, 'A')

		tmpIntro = tmpIntro[end:]
		txtlength = pdf.get_string_width(tmpIntro)
	

	pdf.cell(length, height, tmpIntro, ShowBox, 1, 'A')
	pdf.ln(2)

def title(Text,fontsize,length):
	pdf.ln(2)
	pdf.set_font("Arial", size=fontsize)
	pdf.cell(length, fontsize*Factor, Text, ShowBox, 1, 'A')
	pdf.ln(2)

def makePDF(TMP_FILE):
##############################################################################
	f = open(TMP_FILE,'a+')
	Info = f.readlines()
	f.close()
	Dic = {}
	for line in Info:
		data = line.split()
		Dic[data[0][:-1]] = data[1]

	OUTPUT_DIR = Dic['OUTPUT_DIR']
	CALC_DIR = Dic['CALC_DIR']
	MESH_FILE = Dic['MESH_FILE']
	MATERIAL_DIR = Dic['MATERIAL_DIR']
	PARAM_MOD = Dic['PARAM_MOD']

	Param = __import__(PARAM_MOD)
#	Geom_name = Param.GeomName

################################################################################
	#### Data
	f = open(os.path.splitext(MESH_FILE)[0] + '.dat','r+')
	PreProc = f.readlines()
	f.close()

	MeshDic = {}
	for line in PreProc:
		data = line.split()
		MeshDic[data[0][:-1]] = data[1]

	f = open(OUTPUT_DIR + '/Aster.dat','r+')
	Aster = f.readlines()
	f.close()

	f = open(OUTPUT_DIR + '/PostProc.dat','r+')
	PostProc = f.readlines()
	f.close()

	PPDic = {}
	for line in PostProc:
		data = line.split()
		PPDic[data[0][:-1]] = data[1]

	#### Images
	MeshIm = [OUTPUT_DIR + '/MeshClip.png','Figure 1']
	LaserIm = [OUTPUT_DIR + '/LaserPulse.png','Figure 2']
	FluxIm = [OUTPUT_DIR + '/FluxDist.png','Figure 3']
	SpecIm = [OUTPUT_DIR + '/Capture.png','Figure 4']
	ClSpecIm = [OUTPUT_DIR + '/ClipCapture.png','Figure 5']
	RFIm = [OUTPUT_DIR + '/Rfactor.png','Figure 6']
	
	global pdf
	pdf = FPDF()	
	rmargin = 10
	lmargin = 10
	tmargin = 10
	centre = 105

	Intro = '''This report gives an overview of the key results from the laser flash simulation carried out using Code Aster.''' 
	
	Para1 = '''The simulation was carried out on {0}, which is a disc with radius {1!s} and thickness {2!s}. '''.format(Param.GeomName, Param.Radius,Param.Height1 + Param.Height2)
	if not Param.VoidRadius:
		Para1 = Para1 + "There is no void in this geometry. "
	if Param.VoidRadius:
		Para1 = Para1 + "There is a void in this geometry, which has a radius of {0!s} and a thickness of {1!s}. ".format(Param.VoidRadius, Param.VoidHeight)

	Para1 = Para1+ '''A mesh for this geometry was created using Salome, named Mesh_{5}. The mesh consisits of {0!s} nodes and {1!s} elements, which is made up of {2!s} volumes, {3!s} faces and {4!s} edges. '''.format(MeshDic['Nodes'],MeshDic['Elements'],MeshDic['Volumes'],MeshDic['Faces'],MeshDic['Edges'],Param.GeomName[5:])

	Para2 = '''The laser pulse lasts 0.0008 seconds and imparts 5.366J of energy in to the specimen. Heat flux parameter h is 0. The initial temperature of the specimen is 20. The magnitude of the laser with respect to time can be seen in {0}  while {1} shows the distribution of flux over th top surface and the loads applied at the nodes'''.format(LaserIm[1],FluxIm[1])

	Para3 = '''Results for the specimen are shown below in {0} to {1} '''.format(SpecIm[1],RFIm[1])

	pdf.add_page()
	title('Study Report',20,190)
	paragraph(Intro,12,190)
	title('Sample Geometry',16,190)
	paragraph(Para1,12,190)
	add_image(pdf.get_y(),MeshIm[0],MeshIm[1] + ': Mesh cross-section')
	title('Experiment Conditions',16,190)	
	paragraph(Para2,12,190)
	add_image(pdf.get_y(),LaserIm[0],LaserIm[1] + ': Laser pulse with respect to time')
	add_image(pdf.get_y(),FluxIm[0],FluxIm[1] + ': Heat flux over top surface and applied loads at each node')
	pdf.add_page()
	title('Results',16,190)
	paragraph(Para3,12,190)
	add_image(pdf.get_y(),SpecIm[0],SpecIm[1] + ': Specimen at t  = {0} which is the nearest time step to the half rise time'.format(PPDic['Nearest_Tstep']))
	add_image(pdf.get_y(),ClSpecIm[0],ClSpecIm[1] + ': Specimen cross section at t  = {0} which is the nearest time step to the half rise time'.format(PPDic['Nearest_Tstep']))
	add_image(pdf.get_y(),RFIm[0],RFIm[1] + ': Average temperature on the bottom surface for different radial factors')
	








	pdf.output(CALC_DIR + '/Summary.pdf')
