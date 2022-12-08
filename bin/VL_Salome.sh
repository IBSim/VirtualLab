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
export QT_X11_NO_MITSHM=1
export PATH=/opt/ERMES/ERMES-CPlas-v12.5:/opt/SalomeMeca/V2019.0.3_universal/tools/Code_aster_frontend-20190/bin/:$PATH
cd /home/ibsim/VirtualLab
unset SESSION_MANAGER
source /home/ibsim/miniconda3/etc/profile.d/conda.sh
conda activate VirtualLab
export PYTHONPATH=/home/ibsim/VirtualLab:$PYTHONPATH
python bin/python/Run_Salome_container.py $master $Var $Project $Simulation $ID
exit $?
