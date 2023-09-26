from types import SimpleNamespace as Namespace

###############################
##  GVXR Helix scan example  ##
###############################
import numpy as np
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