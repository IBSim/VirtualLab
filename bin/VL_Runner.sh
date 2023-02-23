#!/bin/bash
set -e
while getopts "m:I:p" options; do
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
export QT_X11_NO_MITSHM=1
export PATH=/opt/ERMES/ERMES-CPlas-v12.5:/opt/SalomeMeca/V2019.0.3_universal/tools/Code_aster_frontend-20190/bin/:$PATH
cd /home/ibsim/VirtualLab
unset SESSION_MANAGER
source /home/ibsim/miniconda3/etc/profile.d/conda.sh
conda activate VirtualLab
export PYTHONPATH=/home/ibsim/VirtualLab:$pypaths:$PYTHONPATH
python bin/python/Run_container.py $pklfile $ID
exit $?
