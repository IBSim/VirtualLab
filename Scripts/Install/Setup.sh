#!/bin/bash

# This should be in VLconfig
VL_DIR_NAME="VirtualLab"
VL_DIR="$HOME/$VL_DIR_NAME"

# Standard update
sudo apt update -y
sudo apt upgrade -y
sudo apt install -y build-essential

# Install python and required packages
sudo apt install -y python3
sudo apt install -y python3-pip
sudo -u ${SUDO_USER:-$USER} pip3 install numpy scipy matplotlib fpdf pillow h5py
sudo -u ${SUDO_USER:-$USER} pip3 install iapws

# Add $VL_DIR to $PYTHONPATH in python env and current shell
if grep -q PYTHONPATH='$PYTHONPATH'$VL_DIR ~/.bashrc; then
  echo "Reference to VirtualLab PYTHONPATH found in .bashrc"
else
  echo "Adding $VL_DIR to PYTHONPATH"
  sudo -u ${SUDO_USER:-$USER} echo 'export PYTHONPATH=$PYTHONPATH'$VL_DIR''  >> ~/.bashrc
  export PYTHONPATH=$PYTHONPATH$VL_DIR
fi


