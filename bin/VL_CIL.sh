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
export QT_QPA_PLATFORM=minimal
cd /home/ibsim/VirtualLab
source /home/user/miniconda/etc/profile.d/conda.sh
conda activate VirtualLab
export PYTHONPATH=/home/ibsim/VirtualLab:$pypath:$PYTHONPATH
conda config --set report_errors false
python bin/python/Run_container.py $pklfile $ID
exit $?
