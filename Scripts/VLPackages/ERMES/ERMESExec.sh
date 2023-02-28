#!/bin/bash
set -e
while getopts "c:f:" options; do
  case "${options}" in
      c)
	  cmd="${OPTARG}"
	  ;;
	  f)
	  filepath="${OPTARG}"
	  ;;
esac
done

dir_name=$(dirname $filepath)
cd $dir_name # certain outputs are written to cwd so make this to the directory of filepath
logfile=$dir_name'/ERMESLog' # log file which results can be written to
$cmd $filepath | tee $logfile

