#!/bin/bash
if [ -f ~/.VLprofile ]; then source ~/.VLprofile; fi

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
sudo ubuntu-drivers autoinstall
sudo apt install -y libcanberra-gtk-module libcanberra-gtk3-module net-tools xterm libopenblas-dev tcl8.5 tk8.5 gfortran libgfortran3 python-tk

### Test to check if salome already exists in current shell's PATH
if hash salome 2>/dev/null; then
  ### If exists, do nothing
  echo "Salome exists in PATH"
  echo "Skipping salome installation"
else
  ### Do more checks
  echo "Salome does not exist in this shell's environment PATH"
  ### Search for reference to salome in ~/.VLprofile
  STRING_TMP="$SALOME_DIR/appli_$SALOME_BIN"
  if [[ $(grep -q "$STRING_TMP" ~/.VLprofile | grep -F -v "#") ]]; then
    echo "Reference to Salome PATH found in .VLprofile"
    echo "Assuming salome is installed"
    echo "Skipping salome installation"
    ### Execute output from grep to try and add to shell's PATH
    source <(grep "STRING_TMP" ~/.VLprofile)
  else
    ### Otherwise download and install
    cd ~
    echo "Salome not found in PATH or ~/.VLprofile"
      if test ! -f "$SALOME_VER".tgz; then
        sudo -u ${SUDO_USER:-$USER} echo "Proceeding to download salome in /home/$USER"
        sudo -u ${SUDO_USER:-$USER} wget https://www.code-aster.org/FICHIERS/"$SALOME_VER".tgz
      fi
    sudo -u ${SUDO_USER:-$USER} echo "Proceeding to unpack salome in /home/$USER"
    sudo -u ${SUDO_USER:-$USER} tar xvf "$SALOME_VER".tgz
    echo "Installing salome in $SALOME_DIR"
    echo -e "$SALOME_DIR\nN" | sudo ./"$SALOME_VER".run
    ### Add to PATH
    echo "Adding salome to PATH"
    sudo -u ${SUDO_USER:-$USER} echo 'export PATH="'$SALOME_DIR'/appli_'$SALOME_BIN':$PATH"'  >> ~/.VLprofile
    export PATH="$SALOME_DIR"/appli_"$SALOME_BIN:$PATH"
    
    ### ~/.bashrc doesn't get read by subshells in ubuntu.
    ### Workaround: store additions to env PATH in ~/.VLprofile & source in bashrc.
    STRING_TMP="if [ -f ~/.VLprofile ]; then source ~/.VLprofile; fi"
    if [[ ! $(grep -F "$STRING_TMP" ~/.bashrc | grep -F -v "#$STRING") ]]; then 
      echo $STRING_TMP >> ~/.bashrc
    fi
    
    ### Test to check if adding to path worked
    if hash salome 2>/dev/null; then
      ### If exists
      echo "Salome now exists in PATH"
      ### ADD TEST HERE TO CHECK WORKING AS EXPECTED
      ### If all working rm salome download files
    else
      ### salome still not installed
      echo "There has been a problem installing salome"
      echo "Check error messages, try to rectify then rerun this script"
    fi
  fi
fi

