#!/bin/bash
set -e
while getopts "c:f:p:" options; do
  case "${options}" in
      c)
	  cmd="${OPTARG}"
	  ;;
	  f)
	  filepath="${OPTARG}"
	  ;;
      p)
	  pypath="${OPTARG}"
	  ;;  
esac
done

export PYTHONPATH=/home/ibsim/VirtualLab:$pypath:$PYTHONPATH

$cmd $filepath



