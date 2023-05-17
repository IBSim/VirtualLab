#!/bin/bash
while getopts "c:d:" options; do
  case "${options}" in
      c)
	    command="${OPTARG}"
      ;;
      d)
	    workingdir="${OPTARG}"
	    ;;

esac
done

export PYTHONPATH=/home/ibsim/VirtualLab:$PYTHONPATH

cd $workingdir
eval $command
exit $?
