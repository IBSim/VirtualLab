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
export PYTHONPATH=/home/ibsim/VirtualLab:$pypath:$PYTHONPATH
# check here that command is a string
eval $command
exit $?
