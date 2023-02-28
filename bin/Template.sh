#!/bin/bash
##################################################
# Template for the script to setup the container #
# You can Setup the enviromant in whatever way   #
# suits your particualr software.
# e.g. seting up/activating conda/virtual envs ect.
# In This case we use various cmd line argumants 
# to pass the required information into the 
# template python script from the Manager to 
# the module.
##################################################
# it's always good practice to use set -e to stop
# in case of error 
set -e
# cmd arguments to get the master, var, project, 
# simulation and container ID from the manager.
while getopts "m:I:p" options; do
  case "${options}" in
      m)
	  pklfile="-m ${OPTARG}"
	  ;;
	  I)
	  ID="-I ${OPTARG}"
	  ;;
      p)
	  pypaths="${OPTARG}"
	  ;;  
esac
done
# We bind the virtualLab code to this directory by default.
cd /home/ibsim/VirtualLab
# if using conda
# source /home/ibsim/miniconda3/etc/profile.d/conda.sh
# conda activate "MY env"
export PYTHONPATH=/home/ibsim/VirtualLab:$pypaths:$PYTHONPATH
# call python
python "python/My_wonderful_python_script.py" $pklfile $ID
# again its good pracrtice to provide a return value in case of error
exit $?
