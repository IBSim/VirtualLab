from types import SimpleNamespace as Namespace

#########################
###### Voxelisation #####
##########################
Vox = Namespace()
# name for the cad2vox run.
# Note: this doubles up as the output file name
Vox.Name = 'Welsh-Dragon'
# name of the mesh(es) you wish to voxelise
Vox.mesh = 'welsh-dragon-small.stl'
# Number of voxels in each dimension
Vox.gridsize = [500,500,500]

#### Optional Arguments #############
# Skip the check for GPU and fallback to CPU
Vox.cpu = True
# use a differnt algorithm to auto fill mesh interior
Vox.solid = False