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
cd /home/ibsim/VirtualLab
source /home/ibsim/miniconda/etc/profile.d/conda.sh
conda activate VirtualLab
export PYTHONPATH=/home/ibsim/VirtualLab:/home/ibsim/GVXR_Install/gvxrWrapper-1.0.5/python3:$PYTHONPATH
conda config --set report_errors false
python Scripts/Common/VLContainer/Run_GVXR_container.py $master $Var $Project $Simulation $ID
exit $?
