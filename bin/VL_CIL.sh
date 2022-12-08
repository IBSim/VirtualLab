#!/bin/bash
while getopts "m:v:p:s:" options; do
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
esac
done
cd /home/ibsim/VirtualLab
source /home/user/miniconda/etc/profile.d/conda.sh
conda activate cil
export PYTHONPATH=/home/ibsim/VirtualLab:$PYTHONPATH
conda config --set report_errors false
python bin/python/Run_CIL_container.py $master $Var $Project $Simulation
exit $?
