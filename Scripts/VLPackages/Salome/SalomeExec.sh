#!/bin/bash
set -e
while getopts "c:f:p:a:r:" options; do
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
	  a)
	  args="${OPTARG}"
	  ;;
	  r)
	  runflag="${OPTARG}"
	  ;;	  
esac
done

export PYTHONPATH=/home/ibsim/VirtualLab:$pypath:$PYTHONPATH

tmpfile=$(mktemp) # make temp file to write port number to

# run salome
$cmd -$runflag --ns-port-log $tmpfile  $filepath args:$args

# get port number from tmpfile and kill salome on that port
port=$(cat $tmpfile)
$cmd kill $port



