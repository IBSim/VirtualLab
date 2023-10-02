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

source /home/ibsim/venv/bin/activate
export PYTHONPATH=$pypath:$PYTHONPATH
# check here that command is a string
eval $command
exit $?
