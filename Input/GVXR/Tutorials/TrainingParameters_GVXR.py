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
# Name of materail file
#GVXR.Material_file
# Position of x-ray beam
GVXR.Beam_PosX = 0
GVXR.Beam_PosY = -100
GVXR.Beam_PosZ = 0
GVXR.Beam_Pos_units = 'mm'
# Bean type must be one of point or parallel
GVXR.beam_type = 'point'
# Beam energy (default units are MeV)
GVXR.energy = 0.08
# postion of the bottom right had corner of the detector in x,y and z
GVXR.Detect_PosX = 0
GVXR.Detect_PosY = 80
GVXR.Detect_PosZ = 0
GVXR.Detect_Pos_units = 'mm'
# number of pixels in x and y, this defines both the resolution of the 
# final images and physical size of te detector plane when combined with spacing_in_mm.
GVXR.Pix_X = 200
GVXR.Pix_Y = 250

#######################
# Optional Parameters #
#######################
# spacing between detector pixels, determines the physical size of the detector.
# default = 0.5 
GVXR.SpacingX = 0.5
GVXR.SpacingY = 0.5
GVXR.Spacing_units='mm'
# Units for beam energy default is 'MeV' can be any of 'eV' 'KeV', 'MeV'
GVXR.energy_units = 'MeV'
# xray beam intensity (no. of x-ray photons) default = 1000
GVXR.Intensity = 1000
# The number if angles you want images from
# (i.e the number of output images) default=180
#GVXR.num_angles = 180
# Determines the rotation agle between each image default=180
# rotation_angle = max_angle / num_angles;
#GVXR.max_angle=180
# String for output image format defualt of None leads to tiff stack
GVXR.image_format='png'
