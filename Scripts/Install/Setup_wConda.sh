#!/bin/bash

# Variables for conda
CONDAVER='Anaconda3-2020.02-Linux-x86_64.sh'
CONDAENV='VirtualLab'

# This should be in VLconfig
VL_DIR_NAME="VirtualLab"
VL_DIR="$HOME/$VL_DIR_NAME"

# Standard update
sudo apt update -y
sudo apt upgrade -y
sudo apt install -y build-essential

# Install conda dependencies
sudo apt install -y libgl1-mesa-glx libegl1-mesa libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6

# Download conda
cd ~
current_dir=$(pwd)
wget https://repo.anaconda.com/archive/"$CONDAVER"
bash $CONDAVER -b -p $HOME/anaconda3
export PATH=$current_dir/anaconda3/bin:$PATH
source ~/.bashrc
# Test conda
if hash conda 2>/dev/null; then
  echo "Conda succesfully installed"
  # rm download if installed
else
  echo "There has been a problem installing salome"
  echo "Check error messages, try to rectify then rerun this script"
fi
#conda --version

# Install conda packages
conda create -n $CONDAENV python -y
conda activate $CONDAENV
conda config --append channels conda-forge
conda install -y numpy scipy matplotlib pillow h5py iapws

# Install python and required packages
sudo apt install -y python3-pip
sudo -u ${SUDO_USER:-$USER} pip3 install fpdf
#sudo -u ${SUDO_USER:-$USER} pip3 install fpdf2

# Add $VL_DIR to $PYTHONPATH in Conda env and current shell
PYV=`python -V`
PYV2=${PYV#* }
PYV=${PYV2%.*}
sudo -u ${SUDO_USER:-$USER} echo $VL_DIR >> $HOME/anaconda3/envs/$CONDAENV/lib/python$PYV/site-packages/VirtualLab.pth
export PYTHONPATH=$PYTHONPATH$VL_DIR

