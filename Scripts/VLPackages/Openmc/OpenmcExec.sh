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


eval $command
