#!/bin/bash
USER_HOME=$(eval echo ~${SUDO_USER})
if [ -f $USER_HOME/.VLprofile ]; then source $USER_HOME/.VLprofile; fi

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $THIS_DIR
source ../../VLconfig.py

### Check if Conda is installed
search_var=anaconda*
conda_dir=$(eval find $USER_HOME -maxdepth 1 -type d -name "$search_var")
if [[ -f $conda_dir/bin/conda ]]; then
  eval "$($conda_dir/bin/conda shell.bash hook)"
else
  search_var=miniconda*
  conda_dir=$(eval find $USER_HOME -maxdepth 1 -type d -name "$search_var")
  if [[ -f $conda_dir/bin/conda ]]; then
    eval "$($conda_dir/bin/conda shell.bash hook)"
  fi
fi

### If conda found activate environment
### If no conda, prerequisites are assumed installed in local python
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

