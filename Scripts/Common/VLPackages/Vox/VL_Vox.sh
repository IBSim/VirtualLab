#!/bin/bash
while getopts "c:p:" options; do
  case "${options}" in
      c)
	  command="${OPTARG}"
	  ;;
      p)
	  pypaths="${OPTARG}"
	  ;;

esac
done
cd /home/ibsim/VirtualLab
source /home/ibsim/miniconda/etc/profile.d/conda.sh
conda activate VirtualLab
export PYTHONPATH=/home/ibsim/VirtualLab:/home/ibsim/GVXR_Install/gvxrWrapper-1.0.5/python3:$pypaths:$PYTHONPATH
conda config --set report_errors false

# check here that command is a string
eval $command

exit $?
