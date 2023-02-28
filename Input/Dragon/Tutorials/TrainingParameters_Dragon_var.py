from types import SimpleNamespace as Namespace

#########################
###### Voxelisation #####
##########################
Vox = Namespace()
# name for the cad2vox run.
# Note: this doubles up as the output file name
Vox.Name = ['Welsh-Dragon','Welsh-Dragon2']
# name of the mesh(es) you wish to voxelise
Vox.mesh = ['welsh-dragon-small.stl','welsh-dragon-small.stl']