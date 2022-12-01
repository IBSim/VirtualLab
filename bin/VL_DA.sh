#!/bin/bash
set -e
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
cd /home/ibsim/VirtualLab
source /home/ibsim/miniconda3/etc/profile.d/conda.sh
conda activate VirtualLab
export PYTHONPATH=/home/ibsim/VirtualLab:$PYTHONPATH
python Scripts/Common/VLContainer/Run_DA_container.py $master $Var $Project $Simulation $ID
exit $?
