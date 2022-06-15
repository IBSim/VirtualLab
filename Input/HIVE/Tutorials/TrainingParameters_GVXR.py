from types import SimpleNamespace as Namespace
GVXR = Namespace()
#############
##  GVXR   ##
#############
#######################
# Required parameters #
#######################
# name for the GVXR run.
GVXR.Name = 'AMAZE'
# name of the mesh(es) you wish to vitually CT scan
GVXR.mesh = 'AMAZE_Sample'
# Name of materail file
GVXR.Material_file = None
# Position of x-ray beam
GVXR.Beam_PosX
GVXR.Beam_PosY
GVXR.Beam_PosZ
# Bean type must be one of point or parallel
GVXR.beam_type = 'point'
# Beam energy (default units are MeV)
GVXR.energy = 0.01
# postion of the bottom right had corner of the detector in x,y and z
GVXR.Detect_PosX
GVXR.Detect_PosY
GVXR.Detect_PosZ
# number of pixels in x and y, this defines both the resolution of the 
# final images and physical size of te detector plane when combined with spacing_in_mm.
GVXR.Pix_X = 200
GVXR.PixY = 250

#######################
# Optional Parameters #
#######################
# spacng betwee detector pixels in mm, determines the physical size of the detector.
# default = 0.5 
GVXR.Spacing_in_mm
# Units for beam energy default='MeV' can be any of 'KeV', 'MeV'
GVXR.energy_units = 'Mev'
# xray beam intensity (no. of x-ray photons) default = 1000
GVXR.Intensity = '1000'
# The number if angles you want images from
# (i.e the number of output images) default=180
#GVXR.num_angles = 180
# Determines the rotation agle between each image default=180
# rotation_angle = max_angle / num_angles;
#GVXR.max_angle=180
# String for output image format defualt of None leads to tiff stack
#GVXR.image_format='png'