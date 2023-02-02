#!/bin/bash
while getopts "m:v:p:s:I:" options; do
  case "${options}" in
      m)
	  master="-m ${OPTARG}"
	  ;;
      v)
	  Var="-v ${OPTARG}"
	  ;;
      p)
	  Project="-p ${OPTARG}"
	  ;;
      s)
	  Simulation="-s ${OPTARG}"
	  ;;
      I)
	  ID="-I ${OPTARG}"
	  ;;
esac
done
export QT_QPA_PLATFORM=minimal
cd /home/ibsim/VirtualLab
source /home/user/miniconda/etc/profile.d/conda.sh
conda activate VirtualLab
export PYTHONPATH=/home/ibsim/VirtualLab:$PYTHONPATH
conda config --set report_errors false
python bin/python/Run_CIL_container.py $master $Var $Project $Simulation $ID
exit $?
