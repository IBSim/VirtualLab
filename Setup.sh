#!/bin/bash

# This should be in VLconfig
VL_DIR="$HOME/VirtualLab"

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
sudo -u ${SUDO_USER:-$USER} echo 'export PATH="'$VL_DIR':$PATH"'  >> ~/.bashrc
export PYTHONPATH=$PYTHONPATH$VL_DIR

