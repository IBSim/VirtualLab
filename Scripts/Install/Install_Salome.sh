#!/bin/bash
set -e
USER_HOME=$(eval echo ~${SUDO_USER})

#########################
### This script is used to install Salome-Meca and its dependencies.
### It first attempts to detect whether it is already installed.
### For VirtualLab, the default config values are as below.
### These can be changed in $VL_DIR/VLconfig_DEFAULT.sh if needed.
###  - Installation location
### SALOME_DIR='/opt/SalomeMeca'
###  - Version number in download filename
### SALOME_VER='salome_meca-2019.0.3-1-universal'
###  - Version number in unpacked directory
### SALOME_BIN='appli_V2019.0.3_universal'
#########################

### Standard update
sudo apt update -y
sudo apt upgrade -y
sudo apt install -y build-essential

### Install salome related libs
#sudo ubuntu-drivers autoinstall
sudo apt install -y libcanberra-gtk-module libcanberra-gtk3-module net-tools xterm libopenblas-dev tcl8.5 tk8.5 gfortran libgfortran3 python-tk dbus

SALOME_DIR="/opt/SalomeMeca"
SALOME_VER="salome_meca-2019.0.3-1-universal"
SALOME_BIN="V2019.0.3_universal"


OS_v=$(eval lsb_release -r -s)

### Test to check if salome already exists in current shell's PATH
if hash salome 2>/dev/null; then
  ### If exists, do nothing
  echo "Salome exists in PATH"
  echo "Skipping salome installation"
else
  ### Otherwise download and install

  ### Checking ubuntu version and applying any specific setup requirements.
  if [[ $OS_v == "20.04" ]]; then
    echo "OS is Ubuntu $OS_v, which doesn't have python2 installed as default."
    echo "Performing some additional steps to install Salome_Meca."
    eval "$($USER_HOME/anaconda3/bin/conda shell.bash hook)"
    if hash conda 2>/dev/null; then
#        if test -d "$USER_HOME/.conda/envs/python2"; then
#          conda activate $USER_HOME/.conda/envs/python2
#        fi
      if test -d "$USER_HOME/anaconda3/envs/python2"; then
        conda activate python2
      fi
    fi
    ### Check python version
    if [[ ! $(python --version 2>&1) == "Python 2"* ]]; then
      echo "You're not running python 2, exiting installation."
      exit 1
    fi
    sudo mkdir $SALOME_DIR
    sudo chown ${SUDO_USER} $SALOME_DIR
    sudo chgrp ${SUDO_USER} $SALOME_DIR
  fi

  cd $USER_HOME
    if test ! -f "$SALOME_VER".tgz; then
      sudo -u ${SUDO_USER:-$USER} echo "Proceeding to download salome in $USER_HOME"
      sudo -u ${SUDO_USER:-$USER} wget https://www.code-aster.org/FICHIERS/"$SALOME_VER".tgz
    fi
  sudo -u ${SUDO_USER:-$USER} echo "Proceeding to unpack salome in $USER_HOME"
  sudo -u ${SUDO_USER:-$USER} tar xvf "$SALOME_VER".tgz

  echo "Installing salome in $SALOME_DIR"
  if [[ $OS_v == "20.04" ]]; then
    echo -e "$SALOME_DIR\nN" | ./"$SALOME_VER".run
  else
    ### Switch in ubuntu v 18->20, need to verify back compatible.
    echo -e "$SALOME_DIR\nN" | sudo ./"$SALOME_VER".run
  fi

  ### Add to PATH
  echo "Adding salome to PATH"
  sudo -u ${SUDO_USER:-$USER} echo 'export PATH="'$SALOME_DIR'/appli_'$SALOME_BIN':$PATH"'  >> $USER_HOME/.bashrc
  export PATH="$SALOME_DIR"/appli_"$SALOME_BIN:$PATH"

  ### Test to check if adding to path worked
  if hash salome 2>/dev/null; then
    ### If exists
    echo "Salome now exists in PATH"
  else
    ### salome still not installed
    echo "There has been a problem installing salome"
    echo "Check error messages, try to rectify then rerun this script"
  fi
fi
