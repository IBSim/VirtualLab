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
while getopts "m:v:p:s:I:" options; do
  case "${options}" in
      m)
	  master="-m ${OPTARG}"
	  ;;
      v)
	  Var="-v ${OPTARG}"
	  ;;
      p)
	  Project="-p ${OPTARG}"
	  ;;
      s)
	  Simulation="-s ${OPTARG}"
	  ;;
	  I)
	  ID="-I ${OPTARG}"
	  ;;
esac
done
# We bind the virtualLab code to this directory by default.
cd /home/ibsim/VirtualLab
# if using conda
# source /home/ibsim/miniconda3/etc/profile.d/conda.sh
# conda activate "MY env"
export PYTHONPATH=/home/ibsim/VirtualLab:$PYTHONPATH
# call python
python "python/My_wonderful_python_script.py" $master $Var $Project $Simulation $ID
# again its good pracrtice to provide a return value in case of error
exit $?
