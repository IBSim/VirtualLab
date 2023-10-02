from types import SimpleNamespace as Namespace
import numpy as np

###############################
##  GVXR Helix scan example  ##
###############################

GVXR = Namespace()
run_names = []
no_slices = 250
top = 100
bottom = -100
# note: slice nums start at 0

for x in range(no_slices):
    run_names.append(f"slice_{x}")
    
GVXR.Name = run_names

# Create an array of all z values for all slices 

GVXR.Model_PosZ = np.linspace(bottom,top,num=no_slices)

# GVXR.Run = [0]*no_slices
# GVXR.Run[0] = 1
# GVXR.Run[100] = 1
# GVXR.Run[249] = 1