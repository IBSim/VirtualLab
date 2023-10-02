from types import SimpleNamespace as Namespace
import VLconfig

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
GVXR.Name = 'AMAZE_360'
# name of the mesh(es) you wish to virtually CT scan
GVXR.mesh = "AMAZE" # full path provided
# Set tube voltage to generate a beam spectrum
GVXR.Energy=[0.8]
GVXR.Intensity = [1000]
# Units for beam energy default is 'MeV' can be any of 'eV' 'KeV', 'MeV'
GVXR.energy_units = 'MeV'
# chemical element corresponding to the material properties
# of each region of the model. Can be any of Full name, Symbol
# or Atomic number (Z). Currently works for every element in the
# peridic table up to Francium, that is elements 1-100
GVXR.Material_list = ["Cu","Cu","W"]
############################################
# Position of x-ray beam
GVXR.Beam_PosX = 0
GVXR.Beam_PosY = -150
GVXR.Beam_PosZ = 0
GVXR.Beam_Pos_units = 'mm'
# Beam type must be one of point or parallel
GVXR.Beam_Type = 'point'

# postion of the bottom right had corner of the detector in x,y and z
GVXR.Detect_PosX = 0
GVXR.Detect_PosY = 80
GVXR.Detect_PosZ = 0
GVXR.Detect_Pos_units = 'mm'
# number of pixels in x and y, this defines both the resolution of the
# final images and physical size of te detector plane when combined with spacing_in_mm.
GVXR.Pix_X = 200
GVXR.Pix_Y = 250
# Postion of center of cad model in x,y and z
GVXR.Model_PosX = 0
GVXR.Model_PosY = 0
GVXR.Model_PosZ = 0
GVXR.Model_Pos_units = 'mm'
GVXR.Model_Mesh_units = 'm'
##############################################
#############################
# Fully Optional Parameters #
#############################
# spacing between detector pixels, determines the physical size of the detector.
# default = 0.5
GVXR.SpacingX = 0.5
GVXR.SpacingY = 0.5
GVXR.Spacing_units='mm'


# The number of angles you want projections
# (i.e the number of output images) default=180
GVXR.num_projections = 360
GVXR.bitrate='int8'
# The rotation angle between each image in degrees default=1
GVXR.angular_step = 1
# String for output image format defualt of None leads to tiff stack
GVXR.image_format = None
# Inital rotation of cad model in degrees about the x,y and z axis
GVXR.rotation = [90,0,0]
GVXR.use_tetra = True
