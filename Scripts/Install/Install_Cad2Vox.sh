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
### CAD2VOX_TAG='Virtual_Lab-V1.0'
### - Decide if you want to build Cad2Vox to use CUDA or just OpenMP
###   Note: the CUDA version is the default and includes both OpenMP and CUDA.
###   Thus you probably only want to change this if you have no intention
###   of using a CUDA GPU and wish to skip installing CUDATOOLKIT.
### CAD2VOX_WITH_CUDA=True
### CUDA_VERSION=11.3
#########################


### Standard update
sudo apt update -y
sudo apt upgrade -y
sudo apt install -y build-essential

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
    if ${USE_CONDA}; then
	conda install cudatoolkit
    else
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
    sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
    wget https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda-repo-ubuntu1804-11-6-local_11.6.0-510.39.01-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu1804-11-6-local_11.6.0-510.39.01-1_amd64.deb
    sudo apt-key add /var/cuda-repo-ubuntu1804-11-6-local/7fa2af80.pub
    sudo apt-get update
    sudo apt-get -y install cuda
    fi
else
    echo "Skiping CUDA install"
fi
# Install GLM, OpenMP and other libraries
sudo apt install -y libglm-dev libgomp1 git mesa-common-dev libglu1-mesa-dev libxi-dev

mkdir -p ${CAD2VOX_DIR}
cd ${CAD2VOX_DIR}

git clone https://github.com/bjthorpe/Cad2vox.git

cd Cad2vox

git checkout ${CAD2VOX_TAG} 

if ${USE_CONDA}; then
    conda install cmake numpy pybind11 tifffile pytest
    conda install -c conda-forge xtensor xtl meshio xtensor-python
else
    pip install -r requirements.txt
    # Build xtl, xtensor and xtensor-python
    mkdir -p libs && cd libs
    #xtl
    git clone https://github.com/xtensor-stack/xtl.git
    cd xtl && cmake && make install && cd ${CAD2VOX_DIR}/Cad2vox/libs
    #xtensor
    git clone https://github.com/xtensor-stack/xtensor.git
    cd xtensor && cmake && make install && cd ${CAD2VOX_DIR}/Cad2vox/libs
    #xtensor-python
    git clone https://github.com/xtensor-stack/xtensor-python.git
    cd xtensor-python && cmake cmake && make install && cd ${CAD2VOX_DIR}/Cad2vox
fi

cd ${CAD2VOX_DIR}/Cad2vox/CudaVox
python3 setup_CudaVox.py install
cd ..
python3 setup_cad2vox.py install

# Run Test Suite
pytest
