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
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
source ${VL_DIR}/VLconfig.py 
GVXR_DIR=${VL_DIR}/third_party/GVXR
mkdir -p ${GVXR_DIR}
cd ${GVXR_DIR}
#install apt packages
sudo apt update
export DEBIAN_FRONTEND=noninteractive
sudo apt install -y linux-headers-generic build-essential subversion libglu1-mesa-dev freeglut3-dev \
mesa-common-dev libfftw3-dev libfftw3-doc zlib1g zlib1g-dev libxrandr-dev \
libxcursor-dev libxinerama-dev libx11-dev libxi-dev libxt-dev python3-tk python3-pip wget unzip
#install latest CMAKE (18.04 repo one is to old)
sudo apt install libssl-dev build-essential -y
wget https://github.com/Kitware/CMake/releases/download/v3.23.1/cmake-3.23.1.tar.gz
tar -xzf cmake-3.23.1.tar.gz
cd cmake-3.23.1
./bootstrap -- -DCMAKE_BUILD_TYPE:STRING=Release
make -j6
# Don't actually make install it as we can run it from the build directory. 
# This avoids conflicting with apt version. 
#make install
cd ${GVXR_DIR}
#build LibTiff
sudo apt install libjpeg-dev liblzma-dev liblz-dev zlib1g-dev -y
wget http://download.osgeo.org/libtiff/tiff-4.0.10.tar.gz
tar -zxvf tiff-4.0.10.tar.gz
cd tiff-4.0.10
./configure
make -j6
make install
ldconfig
cd ${GVXR_DIR}
#build Swig (again 18.04 repo version is too old)
wget https://sourceforge.net/projects/swig/files/swig/swig-4.0.2/swig-4.0.2.tar.gz/download
mv download swig.tar.gz
tar -xzf swig.tar.gz
cd swig-4.0.2
sudo apt install libpcre3 libpcre3-dev -y
./configure
make
make install
cd ${GVXR_DIR}
# install python packages
if ${USE_CONDA}; then
    conda install matplotlib scikit-image pydantic numexpr
else
    sudo -u ${SUDO_USER:-$USER} pip3 install matplotlib scikit-image pydantic numexpr
fi
#conda install scikit-image
#grab the GVXR Source
wget https://sourceforge.net/projects/gvirtualxray/files/1.1/gVirtualXRay-1.1.3-Source.zip/download
mv download gVirtualXRay-1.1.3-Source.zip
unzip gVirtualXRay-1.1.3-Source.zip
cd gVirtualXRay-1.1.3
mkdir -p bin-release
export GVXR_INSTALL_DIR=${GVXR_DIR}_Install

pwd
cd bin-release
${GVXR_DIR}/cmake-3.23.1/bin/cmake -DCMAKE_BUILD_TYPE:STRING=Release \
-DCMAKE_INSTALL_PREFIX:STRING=$GVXR_INSTALL_DIR \
-DBUILD_TESTING:BOOL=OFF \
-DBUILD_WRAPPER_CSHARP:BOOL=OFF \
-DBUILD_WRAPPER_JAVA:BOOL=OFF \
-DBUILD_WRAPPER_OCTAVE:BOOL=OFF \
-DBUILD_WRAPPER_PERL:BOOL=OFF \
-DBUILD_WRAPPER_PYTHON3:BOOL=ON \
-DBUILD_WRAPPER_R:BOOL=OFF \
-DBUILD_WRAPPER_RUBY:BOOL=OFF \
-DBUILD_WRAPPER_TCL:BOOL=OFF \
-DUSE_LIBTIFF:BOOL=OFF \
-S ${GVXR_DIR}/gVirtualXRay-1.1.3 -B $PWD

# now one final make build GVXR.
make -j6
make install
echo "Adding GVXR to PYTHONPATH"
sudo echo "export PYTHONPATH=${GVXR_INSTALL_DIR}/gvxrWrapper-1.0.6/python3:\${PYTHONPATH}" >> $USER_HOME/.VLprofile
source ${USER_HOME}/.VLprofile
echo "Installing Speckpy"
cd ${VL_DIR}/third_party
git clone https://bitbucket.org/spekpy/spekpy_release.git
cd spekpy_release
pip install .
cd ${VL_DIR}
cleanup
