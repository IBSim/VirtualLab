#!/bin/bash
USER_HOME=$(eval echo ~${SUDO_USER})
if [ -f $USER_HOME/.VLprofile ]; then source $USER_HOME/.VLprofile; fi

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $THIS_DIR
source ../../VLconfig.py

### Check if Conda is installed, if so activate environment
### If no conda, prerequisites are assumed installed in local python
eval "$($HOME/anaconda3/bin/conda shell.bash hook)"
if hash conda 2>/dev/null; then
  CONDAENV="$(basename -- $VL_DIR)"
  conda activate $CONDAENV
fi

### Build VirtualLab documentation using sphinx
echo "Building documentation"
cd $VL_DIR/docs
make clean
make html
sudo chown ${SUDO_USER} -R $VL_DIR/docs/build
sudo chgrp ${SUDO_USER} -R $VL_DIR/docs/build
cd $VL_DIR

if test -f $VL_DIR/docs.html; then
  sudo rm docs.html
fi
sudo -u ${SUDO_USER:-$USER} ln -s docs/build/html/index.html docs.html
echo
echo "A shortcut to the documentation (docs.html) has been created in $VL_DIR."
