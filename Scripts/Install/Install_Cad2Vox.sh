#!/bin/bash

USER_HOME=$(eval echo ~${SUDO_USER})
if [ -f $USER_HOME/.VLprofile ]; then source $USER_HOME/.VLprofile; fi
#########################
### This script is used to install Cad2Vox and its dependencies.
### 
### For VirtualLab, the default config values are as below.
### These can be changed in $VL_DIR/VLconfig_DEFAULT.sh if needed.
###  - Installation location
### CAD2VOX_DIR='$HOME/VirtualLab/Cad2Vox'
###  - Git tag, used to identify where to pull from within
###    the Cad2Vox git Repo. 
### CAD2VOX_TAG='VirtualLab_V1.55'
### - Decide if you want to build Cad2Vox to use CUDA or just OpenMP
###   Note: the CUDA version is the default and includes both OpenMP and CUDA.
###   Thus you probably only want to change this if you have no intention
###   of using a CUDA GPU and wish to skip installing CUDATOOLKIT.
### CAD2VOX_WITH_CUDA=True
### CUDA_VERSION=11.3
#########################


### Standard update
export DEBIAN_FRONTEND=noninteractive
sudo apt update -y
sudo apt upgrade -y
sudo apt install -y build-essential cmake python3-pybind11

source ${VL_DIR}/VLconfig.py

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

if ${CAD2VOX_WITH_CUDA}; then
    echo "Installing CUDA"
wget https://developer.download.nvidia.com/compute/cuda/11.4.4/local_installers/cuda_11.4.4_470.82.01_linux.run
sudo sh cuda_11.4.4_470.82.01_linux.run --silent
sudo rm cuda_11.4.4_470.82.01_linux.run
else
    echo "Skiping CUDA install"
fi
# Install GLM, OpenMP and other libraries
sudo apt install -y libglm-dev libgomp1 git mesa-common-dev libglu1-mesa-dev libxi-dev

cd ${VL_DIR}

git clone https://github.com/bjthorpe/Cad2vox.git
sudo chown ${USER}:${USER} Cad2vox/*
cd Cad2vox

git checkout ${CAD2VOX_TAG} 

if ${USE_CONDA}; then
    conda install -y cmake numpy pybind11 tifffile pytest pillow pandas
    conda install -y -c conda-forge xtensor xtl meshio xtensor-python
else
    pip3 install --user -r requirements.txt
    # Build xtl, xtensor and xtensor-python
    mkdir -p libs && cd libs
    #xtl
    git clone https://github.com/xtensor-stack/xtl.git
    sudo chown ${SUDO_USER:-$USER} xtl/*
    cd xtl && cmake . && sudo make install && cd ${CAD2VOX_DIR}/libs
    #xtensor
    git clone https://github.com/xtensor-stack/xtensor.git
     sudo chown ${SUDO_USER:-$USER} xtensor/*
    cd xtensor && cmake . && sudo make install && cd ${CAD2VOX_DIR}/libs
    #xtensor-python
    git clone https://github.com/xtensor-stack/xtensor-python.git
    sudo chown ${SUDO_USER:-$USER} xtensor-python/*
    cd xtensor-python && cmake . && sudo make install && cd ${CAD2VOX_DIR}
fi

cd ${CAD2VOX_DIR}/CudaVox
python3 -m pip install --user .
cd ${CAD2VOX_DIR}
python3 -m pip install --user .

# Run Test Suite
if ${CAD2VOX_WITH_CUDA}; then
pytest
else
pytest -K "not CUDA"
fi
