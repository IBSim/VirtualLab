#!/bin/bash --login
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
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate coms_test
export PYTHONPATH=/home/ibsim/VirtualLab:$PYTHONPATH
python bin/python/Run_Test_container.py $master $Var $Project $Simulation $ID
exit $?
