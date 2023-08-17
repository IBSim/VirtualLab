from types import SimpleNamespace as Namespace
GVXR = Namespace()
#############
##  GVXR   ##
#############
#######################
# Required parameters #
#######################
############################################################
# Note for real space quantites units can be any off: 
# "um", "micrometre", "micrometer", "mm", "millimetre", 
# "millimeter", "cm", "centimetre", "centimeter", "dm",
# "decimetre", "decimeter", "m", "metre", "meter", "dam",
# "decametre", "decameter", "hm", "hectometre", "hectometer",
#  "km", "kilometre", "kilometer"
#
# However the default is 'mm'
############################################################
# name for the GVXR run.
GVXR.Name = 'AMAZE'
# name of the mesh(es) you wish to vitually CT scan
GVXR.mesh = 'AMAZE_Sample'
# Flag to use spekpy to gentrate a beam spectrum
GVXR.use_spekpy = True
# Beam energy (default units are MeV)
GVXR.Energy = [0.08]
# xray beam intensity (no. of x-ray photons)
GVXR.Intensity = [1000]
GVXR.Tube_Voltage = 80
GVXR.Tube_Angle = 12
############################################
# Nikon parameter input file
# Use paramters read from .xtekct file
GVXR.Nikon_file = '/home/ben/test.xtek'
############################################
############################################
# Optional parmeters when using Nikon file #
############################################
# Anything you define from this block will 
# override the values read in if using a 
# .xtekct file. 
# Note: thease are required prameters when 
# NOT USING a .xtekct file.
############################################
# Position of x-ray beam
#GVXR.Beam_PosX = 0
#GVXR.Beam_PosY = -100
#GVXR.Beam_PosZ = 0
#GVXR.Beam_Pos_units = 'mm'
# Beam type must be one of point or parallel
#GVXR.beam_type = 'point'

# postion of the bottom right had corner of the detector in x,y and z
#GVXR.Detect_PosX = 0
#GVXR.Detect_PosY = 80
#GVXR.Detect_PosZ = 0
#GVXR.Detect_Pos_units = 'mm'
# number of pixels in x and y, this defines both the resolution of the 
# final images and physical size of te detector plane when combined with spacing_in_mm.
#GVXR.Pix_X = 200
#GVXR.Pix_Y = 250
# Postion of center of cad model in x,y and z
#GVXR.Model_PosX = 0
#GVXR.Model_PosY = 0
#GVXR.Model_PosZ = 0
#GVXR.Model_Pos_units = 'mm'
##############################################
#############################
# Fully Optional Parameters #
#############################
# Name of materail file
#GVXR.Material_file = 'materials.csv'
# spacing between detector pixels, determines the physical size of the detector.
# default = 0.5 
#VXR.SpacingX = 0.5
#GVXR.SpacingY = 0.5
#GVXR.Spacing_units='mm'
# Units for beam energy default is 'MeV' can be any of 'eV' 'KeV', 'MeV'
#GVXR.energy_units = 'MeV'

# The number of angles you want projections
# (i.e the number of output images) default=180
#GVXR.num_projections = 180
# The rotation angle between each image in degrees default=1
#GVXR.angular_step = 1 
# String for output image format defualt of None leads to tiff stack
#GVXR.image_format = 'png'
# Inital rotation of cad model about the x,y and z axis
#GVXR.model_rotation = [0,0,0]