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
export GVXR_INSTALL_DIR=${GVXR_DIR}_Install
mkdir -p ${GVXR_DIR}
cd ${GVXR_DIR}
#install apt packages
sudo apt update
export DEBIAN_FRONTEND=noninteractive
sudo apt install -y linux-headers-generic build-essential subversion libglu1-mesa-dev freeglut3-dev \
mesa-common-dev libfftw3-dev libfftw3-doc zlib1g zlib1g-dev libxrandr-dev \
libxcursor-dev libxinerama-dev libx11-dev libxi-dev libxt-dev python3-tk python3-pip wget unzip curl 
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
#install latest CMAKE and swig from brew (18.04 repo one is to old)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
echo 'eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"' >> /home/ibsim/.profile
eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"
brew install cmake swig
cd ${GVXR_DIR}
# install python packages
if ${USE_CONDA}; then
    conda install matplotlib scikit-image pydantic numexpr
else
    sudo -u ${SUDO_USER:-$USER} pip3 install matplotlib scikit-image pydantic numexpr
fi
#conda install scikit-image
#grab the GVXR Source
wget https://sourceforge.net/projects/gvirtualxray/files/1.1/gVirtualXRay-1.1.5-Source.zip/download
mv download gVirtualXRay-1.1.5-Source.zip
unzip gVirtualXRay-1.1.5-Source.zip
cd gVirtualXRay-1.1.5
mkdir -p bin-release

cd bin-release
cmake -DCMAKE_BUILD_TYPE:STRING=Release \
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
-DCMAKE_POLICY_DEFAULT_CMP0072=NEW \
-S .. -B $PWD

# Bodge to fix ubuntu specifc issues
# in this case we build glew first then copy the resluting libaries
# to the "correct" place as cmake put them in lib under Ubuntu. Whereas 
# under all other unix platforms it is in lib64. This cases linking 
# errors with swig which have yet to be addressed by Frank.
make assimp -j12
cmake -DCMAKE_BUILD_TYPE:STRING=Release \
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
-DCMAKE_POLICY_DEFAULT_CMP0072=NEW \
-S .. -B $PWD
make glew -j12
mkdir -p gvxr/glew-install/lib64
cp gvxr/glew-install/lib/lib*.a gvxr/glew-install/lib64
make glfw -j12
mkdir -p glfw-install/lib64
cp glfw-install/lib/lib*.a glfw-install/lib64
make gVirtualXRay -j12
make SimpleGVXR -j12
make gvxrPython3 -j12

# now one final make to link the rest of GVXR as normal
make -j12
make install
brew install glfw glew
echo "Adding GVXR to PYTHONPATH"
sudo echo "export PYTHONPATH=${GVXR_INSTALL_DIR}/gvxrWrapper-1.0.5/python3:\${PYTHONPATH}" >> $USER_HOME/.VLprofile
source ${USER_HOME}/.VLprofile
echo "Installing Speckpy"
cd ${VL_DIR}/third_party
git clone https://bitbucket.org/spekpy/spekpy_release.git
cd spekpy_release
pip install .
cd ${VL_DIR}
cleanup
