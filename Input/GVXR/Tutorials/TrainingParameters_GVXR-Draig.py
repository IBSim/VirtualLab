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
GVXR.Name = 'Dragon'
# name of the mesh(es) you wish to vitually CT scan
GVXR.mesh = 'welsh-dragon-small.stl'
# chemical element corresponding to the material properties
# of each region of the model. Can be any of Full name, Symbol
# or Atomic number (Z). Currently works for every element in the
# periodic table up to Francium, that is elements 1-100.
# Note: in this case we only supply one element since .stl 
# files have only one region. Thus we define one material 
# for the whole dragon model.
GVXR.Material_list = ["Al"]
# Beam energy (default units are MeV)
GVXR.Energy = 300
# xray beam intensity (no. of x-ray photons)
GVXR.Intensity = 1000
# Position of x-ray beam
GVXR.Beam_PosX = 0
GVXR.Beam_PosY = -250
GVXR.Beam_PosZ = 0
GVXR.Beam_Pos_units = 'mm'
# Beam type must be one of point or parallel
GVXR.Beam_Type = 'point'

# postion of the bottom right had corner of the detector in x,y and z
GVXR.Detect_PosX = 0
GVXR.Detect_PosY = 110
GVXR.Detect_PosZ = 0
GVXR.Detect_Pos_units = 'mm'
# number of pixels in x and y, this defines both the resolution of the 
# final images and physical size of te detector plane when combined with spacing_in_mm.
GVXR.Pix_X = 300
GVXR.Pix_Y = 350
# Postion of center of cad model in x,y and z
GVXR.Model_PosX = 0
GVXR.Model_PosY = 0
GVXR.Model_PosZ = 0
GVXR.Model_Pos_units = 'mm'
##############################################
#############################
# Fully Optional Parameters #
#############################
# spacing between detector pixels, determines the physical size of the detector.
# default = 0.5 
GVXR.SpacingX = 0.7
GVXR.SpacingY = 0.7
GVXR.Spacing_units='mm'
# Units for beam energy default is 'MeV' can be any of 'eV' 'KeV', 'MeV'
GVXR.energy_units = 'MeV'

# String for output image format default is tiff
GVXR.image_format = 'png'
# set the output image bitrate can be any off int8, int16 or float32
GVXR.Im_bitrate = 'int8'
# Initial rotation of cad model in degrees about the x,y and z axis
GVXR.rotation = [0,0,0]
# flat field normalize output image(s)
GVXR.FFNorm = True
