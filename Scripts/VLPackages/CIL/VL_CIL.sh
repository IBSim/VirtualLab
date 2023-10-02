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
export QT_QPA_PLATFORM=minimal

source /home/user/miniconda/etc/profile.d/conda.sh
conda activate VirtualLab
export PYTHONPATH=$pypath:$PYTHONPATH
conda config --set report_errors false
# check here that command is a string
eval $command
exit $?
