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
	  args="args:${OPTARG}"
	  ;;
	  r)
	  runflag="${OPTARG}"
	  ;;	  
esac
done

export PYTHONPATH=$pypath:$PYTHONPATH

tmpfile=$(mktemp) # make temp file to write port number to

# run salome
$cmd -$runflag --ns-port-log $tmpfile  $filepath $args

if [ $runflag = 't' ] ; then
	# get port number from tmpfile and kill salome on that port (only needed if GUI hasnt been opened)
	port=$(cat $tmpfile)
	$cmd kill $port
else 
	sleep 3 #  gives a chance for salome gui to clean up (otherwise errors are printed to screen)
fi


