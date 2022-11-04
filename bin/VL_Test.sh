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
export PYTHONPATH=/home/ibsim/VirtualLab:$PYTHONPATH
python3 Scripts/Common/VLContainer/Run_Test_container.py $master $Var $Project $Simulation $ID
exit $?
