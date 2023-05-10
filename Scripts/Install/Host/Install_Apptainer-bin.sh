#!/bin/bash

set -e
USER_HOME=$(eval echo ~${SUDO_USER})
if [ -f $USER_HOME/.VLprofile ]; then source $USER_HOME/.VLprofile; fi

#########################
### This script is used to install Apptainer and its dependencies.
### It first attempts to detect whether it is already installed.
#########################

### Standard update
sudo apt update -y
sudo apt upgrade -y
sudo apt install -y build-essential

#source "$VL_DIR/VLconfig.py" # Enables this script to be run seperately
if [ -f $VL_DIR/VLconfig.py ]; then source $VL_DIR/VLconfig.py; fi

### Test to check if Apptainer already exists in current shell's PATH
if hash apptainer 2>/dev/null; then
  ### If exists, do nothing
  echo "Apptainer exists in PATH"
  echo "Skipping Apptainer installation"
else
  sudo apt update
  sudo apt install -y software-properties-common
  sudo add-apt-repository -y ppa:apptainer/ppa
  sudo apt update
  sudo apt install -y apptainer

  ### Test to check if installation worked
  if hash apptainer 2>/dev/null; then
    ### If exists
    echo "Apptainer has been installed"
  else
    ### git still not installed
    echo "There has been a problem installing Apptainer"
    echo "Check error messages, try to rectify then rerun this script"
    exit
  fi
fi
