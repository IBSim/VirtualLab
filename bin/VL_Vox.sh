#!/bin/bash
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
source /home/ibsim/miniconda/etc/profile.d/conda.sh
conda activate VirtualLab
export PYTHONPATH=/home/ibsim/VirtualLab:/home/ibsim/GVXR_Install/gvxrWrapper-1.0.5/python3:$pypaths:$PYTHONPATH
conda config --set report_errors false
python bin/python/Run_container.py $pklfile $ID
exit $?
