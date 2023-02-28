#!/bin/bash --login
while getopts "m:v:p:s:I:" options; do
  case "${options}" in
      m)
	  pklfile="-m ${OPTARG}"
	  ;;
	  I)
	  ID="-I ${OPTARG}"
	  ;;
      p)
	  pypaths="${OPTARG}"
	  ;; 
esac
done
cd /home/ibsim/VirtualLab
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate coms_test
export PYTHONPATH=/home/ibsim/VirtualLab:$pypaths:$PYTHONPATH
python bin/python/Run_container.py $pklfile $ID
exit $?
