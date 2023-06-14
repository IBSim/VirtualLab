from types import SimpleNamespace as Namespace
ImPreProc = Namespace()
# path to experimental data either absolute or name of file 
# in Input/$PROJECT}/${SIMULATION}/Data
ImPreProc.Exp_Data = '/path/to/raw/data.vol'
# path to reconstructed GVXR data if needed. Either 
# absolute path or or name of file 
# in Output/$PROJECT}/${SIMULATION}/CIL_Images
ImPreProc.Sim_Data = None


###################################################
# Normalization parameters
##################################################
# image dims
ImPreProc.img_dim_x = 1414
ImPreProc.img_dim_y = 1417
ImPreProc.num_slices = 1417
# image data type default is unsigned 16bit i.e. 'u2'
ImPreProc.raw_dtype = 'u2'
# No. of histogram bins default is 512
# ImPreProc.nbins = 512
# percentage of max pixel value to use as background 
# default is 0.1 (i.e. 10%) 
# ImPreProc.des_bg = 0.1
# percentage of max pixel value to use as foreground 
# (that is the material) default is 0.9 (i.e. 90%) 
# ImPreProc.des_fg = 0.9
# pixel width of the peaks used, default is 1
# ImPreProc.peak_width = 1
# sets number of peaks that are used for 
# comparison when determining air vs material.
# default is 10
# ImPreProc.set_order_no = 10
# bool to set if you want to keep the exp data as a .vol file
# otherwise outputs a tiff stack
# ImPreProc.Keep_Raw  = True

###################################################
# Registration parameters
##################################################
# location of cad2vox data to optionally use as mask
# ImPreProc.Vox_Data = None
# no. of iterations
# ImPreProc.Iterations = 5000
# number of random samples
# ImPreProc.Samples = 5000
# no of internal resolutions to use
# ImPreProc.Resolutions = 4
# path to parameters file to control
# literally everything else. One file is used for each 
# type of transform. 
# There are loads of things you can tweak with these.
# Example files of params I found worked well are in 
# Scripts/VLPackages/ImPreProc/Default_ImPreProc.
# more info and examples can be found on the ITKelastix github:
# https://github.com/InsightSoftwareConsortium/ITKElastix
# there are also a number of examples on the elastix
# modelZoo here: https://github.com/SuperElastix/ElastixModelZoo
# ImPreProc.Reg_Params