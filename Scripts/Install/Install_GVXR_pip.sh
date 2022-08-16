#!/bin/bash
set -e
USER_HOME=$(eval echo ~${SUDO_USER})
if [ -f $USER_HOME/.VLprofile ]; then source $USER_HOME/.VLprofile; fi
#########################
### This script is used to install GVXR and its dependencies.
### 
### For VirtualLab, the default config values are as below.
### These can be changed in $VL_DIR/VLconfig_DEFAULT.sh if needed.
###  - Installation location
### GVXR_DIR='$VL_DIR/third_party/GVXR'
#########################
# function to fix ownership issues caused by running everything with Sudo
cleanup() {
# Vlprofile should be owned by the user not root
sudo chown ${SUDO_USER} $USER_HOME/.VLprofile
# everything in VirtualLab directory should be owned by the user not root
sudo chown ${SUDO_USER} -R ${VL_DIR}
# to add: make sure pip and conda ownership issues are fixed if needed.
}
########################
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
  USE_CONDA=true
  CONDAENV="$(basename -- $VL_DIR)"

  if conda info --envs | grep -q $CONDAENV; then
      echo "Found existing VirtualLab Conda environment"      
  else
      echo "VirtualLab conda environment not found so creating."
      conda create -n $CONDAENV
  fi
  conda activate $CONDAENV

else
    USE_CONDA=false
fi
#################################################################
source ${VL_DIR}/VLconfig.py 
# install python packages
if ${USE_CONDA}; then
    conda install matplotlib scikit-image pydantic
else
    sudo -u ${SUDO_USER:-$USER} pip3 install matplotlib scikit-image pydantic
fi
sudo -u ${SUDO_USER:-$USER} env "PATH=$PATH" pip install numexpr
#install speckpy
echo "Installing Speckpy"
cd ${VL_DIR}/third_party
git clone https://bitbucket.org/spekpy/spekpy_release.git
cd spekpy_release
sudo -u ${SUDO_USER:-$USER} env "PATH=$PATH" pip install .
# install GVXR pip version
sudo -u ${SUDO_USER:-$USER} env "PATH=$PATH" pip install GVXR
cd ${VL_DIR}
cleanup
