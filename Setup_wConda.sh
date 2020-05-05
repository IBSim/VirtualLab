#!/bin/bash

# Variables for conda
CONDAVER='Anaconda3-2020.02-Linux-x86_64.sh'
CONDAENV='VirtualLab'

# Variables for salome
# Installation location
SALOMEDIR='/opt/SalomeMeca'
# Version number in download filename
SALOMEVER='salome_meca-2019.0.3-1-universal'
# Version number in unpacked directory
SALOMEBIN='appli_V2019.0.3_universal'

# This should be in VLconfig
VL_DIR="$HOME/VirtualLab"

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
#echo -e "$SALOMEDIR\nN" | sudo ./"$SALOMEVER".run
#echo -e "\nqYes\n\nYes\n" | bash $CONDAVER
#source ~/.bashrc
bash $CONDAVER -b -p $HOME/anaconda3
export PATH=$current_dir/anaconda3/bin:$PATH
source ~/.bashrc
# Test conda
#[ hash conda 2>/dev/null ] && echo "Conda succesfully installed"
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
#sudo apt install -y python3
sudo apt install -y python3-pip
#sudo -u ${SUDO_USER:-$USER} pip3 install numpy scipy matplotlib fpdf pillow h5py iapws
sudo -u ${SUDO_USER:-$USER} pip3 install fpdf
#sudo -u ${SUDO_USER:-$USER} pip3 install fpdf2

# Add $VL_DIR to $PYTHONPATH in Conda env and current shell
PYV=`python -V`
PYV2=${PYV#* }
PYV=${PYV2%.*}
sudo -u ${SUDO_USER:-$USER} echo $VL_DIR >> $HOME/anaconda3/envs/$CONDAENV/lib/python$PYV/site-packages/VirtualLab.pth
export PYTHONPATH=$PYTHONPATH$VL_DIR

# Install salome related libs
sudo ubuntu-drivers autoinstall
#sudo apt install -y libcanberra-gtk-module libcanberra-gtk3-module
#sudo apt install -y net-tools
#sudo apt install -y xterm
#sudo apt install -y libopenblas-dev
#sudo apt install -y tcl8.5
#sudo apt install -y tk8.5
#sudo apt install -y gfortran
#sudo apt install -y libgfortran3
#sudo apt install -y python-tk

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
  if grep -q "$SALOMEDIR/$SALOMEBIN" ~/.bashrc; then
    echo "Reference to Salome PATH found in .bashrc"
    echo "Assuming Salome is installed"
    echo "Skipping Salome installation"
    # Execute output from grep to try and add to shell's PATH
    source <(grep "$SALOMEDIR/$SALOMEBIN" ~/.bashrc)
  else
    # Otherwise download and install
    echo "Salome not found in PATH or .bashrc"
    echo "Proceeding to download and unpack salome in /home/$USER"
    cd ~
    sudo -u ${SUDO_USER:-$USER} wget https://www.code-aster.org/FICHIERS/"$SALOMEVER".tgz
    sudo -u ${SUDO_USER:-$USER} tar xvf "$SALOMEVER".tgz
    echo "Installing salome in $SALOMEDIR"
    echo -e "$SALOMEDIR\nN" | sudo ./"$SALOMEVER".run
    # Add to PATH
    echo "Adding salome to PATH"
    sudo -u ${SUDO_USER:-$USER} echo 'export PATH="'$SALOMEDIR'/'$SALOMEBIN':$PATH"'  >> ~/.bashrc
    sudo -u ${SUDO_USER:-$USER} echo 'export PATH="'$SALOMEDIR'/'$SALOMEBIN':$PATH"'  >> ~/VirtualLab/.VLprofile
    export PATH="$SALOMEDIR"/"$SALOMEBIN:$PATH"
    
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
