from types import SimpleNamespace as Namespace
GVXR = Namespace()
#############
##  GVXR   ##
#############
#######################
# Required parameters #
#######################
# name for the GVXR run.
GVXR.Name = 'Dragon_Nikon'
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
GVXR.Energy = [0.08]
# xray beam intensity (no. of x-ray photons)
GVXR.Intensity = [1000]
# Units for beam energy default is 'MeV' can be any of 'eV' 'KeV', 'MeV'
GVXR.energy_units = 'MeV'
# Initial rotation of cad model in degrees about the x,y and z axis
GVXR.rotation = [0,0,90]
GVXR.bitrate='int8'
#Name of .xect Nikonfile to read remaining info from
GVXR.Nikon_file = 'Welsh_Dragon.xtekct'
