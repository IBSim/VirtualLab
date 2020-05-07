#!/bin/bash

### Default values to be replaced by VLconfig
#VL_DIR_NAME="VirtualLab"
#VL_DIR="$HOME/$VL_DIR_NAME"

### Variables for salome
### These should now be read from VLconfig.py
### Installation location
#SALOMEDIR='/opt/SalomeMeca'
### Version number in download filename
#SALOMEVER='salome_meca-2019.0.3-1-universal'
### Version number in unpacked directory
#SALOMEBIN='appli_V2019.0.3_universal'

# Standard update
sudo apt update -y
sudo apt upgrade -y
sudo apt install -y build-essential

# Install salome related libs
sudo ubuntu-drivers autoinstall
sudo apt install -y libcanberra-gtk-module libcanberra-gtk3-module net-tools xterm libopenblas-dev tcl8.5 tk8.5 gfortran libgfortran3 python-tk

# Test to check if salome already exists in current shell's PATH
if hash salome 2>/dev/null; then
  # If exists, do nothing
  echo "Salome exists in PATH"
  echo "Skipping salome installation"
else
  # Do more checks
  echo "Salome does not exist in this shell's environment PATH"
  # Search for reference to salome in .bashrc
  if grep -q "$SALOMEDIR/appli_$SALOMEBIN" ~/.bashrc; then
    echo "Reference to Salome PATH found in .bashrc"
    echo "Assuming salome is installed"
    echo "Skipping salome installation"
    # Execute output from grep to try and add to shell's PATH
    source <(grep "$SALOMEDIR/appli_$SALOMEBIN" ~/.bashrc)
  else
    # Otherwise download and install
    cd ~
    echo "Salome not found in PATH or .bashrc"
      if test ! -f "$SALOMEVER".tgz; then
        echo "Proceeding to download salome in /home/$USER"
        sudo -u ${SUDO_USER:-$USER} wget https://www.code-aster.org/FICHIERS/"$SALOMEVER".tgz
      fi
    echo "Proceeding to unpack salome in /home/$USER"
    sudo -u ${SUDO_USER:-$USER} tar xvf "$SALOMEVER".tgz
    echo "Installing salome in $SALOMEDIR"
    echo -e "$SALOMEDIR\nN" | sudo ./"$SALOMEVER".run
    # Add to PATH
    echo "Adding salome to PATH"
    sudo -u ${SUDO_USER:-$USER} echo 'export PATH="'$SALOMEDIR'/appli_'$SALOMEBIN':$PATH"'  >> ~/.bashrc
    #sudo -u ${SUDO_USER:-$USER} echo 'export PATH="'$SALOMEDIR'/'$SALOMEBIN':$PATH"'  >> ~/VirtualLab/.VLprofile
    export PATH="$SALOMEDIR"/appli_"$SALOMEBIN:$PATH"
    
    # Test to check if adding to path worked
    if hash salome 2>/dev/null; then
      # If exists
      echo "Salome now exists in PATH"
      # ADD TEST HERE TO CHECK WORKING AS EXPECTED
      # If all working rm salome download files
    else
      # salome still not installed
      echo "There has been a problem installing salome"
      echo "Check error messages, try to rectify then rerun this script"
    fi
  fi
fi

