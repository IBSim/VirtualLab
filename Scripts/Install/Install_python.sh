#!/bin/bash

### Default values to be replaced by VLconfig
#VL_DIR_NAME="VirtualLab"
#VL_DIR="$HOME/$VL_DIR_NAME"

### Variables for conda
### $CONDAVER should now be read from VLconfig.py
### This will not work for all cases, but does from Install_VirtualLab.sh
### Need to code more robustly
#CONDAVER='Anaconda3-2020.02-Linux-x86_64.sh'
CONDAENV=$VL_DIR_NAME
### By default don't install conda unless triggered by flag
CONDA_INST="n"

### Get flags to install python locally or in conda env.
usage() {
  echo "Usage: $0 [ -C {y/n} ]" 1>&2 
}
exit_abnormal() {
  usage
  exit 1
}
while getopts ":C:" options; do 
  case "${options}" in
    C) # If C option triggered
      CONDA_INST=${OPTARG}
      if [ "$CONDA_INST" == "y" ]; then
        echo "Attempting to instal VirtualLab within a Conda env"
      elif [ "$CONDA_INST" == "n" ]; then
        echo "Attempting to instal VirtualLab using local python"
      else
        echo "Error: Invalid option argument $CONDA_INST" >&2
        exit_abnormal
      fi
      ;;
    :)  # If expected argument omitted:
      echo "Error: Option -${OPTARG} requires an argument."
      exit_abnormal
      ;;
    *)  # If unknown (any other) option:
      echo "Error: Invalid option -$OPTARG" >&2
      exit_abnormal
      ;;
  esac
done

# Standard update
sudo apt update -y
sudo apt upgrade -y
sudo apt install -y build-essential

if [ "$CONDA_INST" == "n" ]; then
  # Install python and required packages
  sudo apt install -y python3
  sudo apt install -y python3-pip
  sudo -u ${SUDO_USER:-$USER} pip3 install numpy scipy matplotlib fpdf pillow h5py
  sudo -u ${SUDO_USER:-$USER} pip3 install iapws

  # Add $VL_DIR to $PYTHONPATH in python env and current shell
  if grep -q PYTHONPATH='$PYTHONPATH'$VL_DIR ~/.bashrc; then
    echo "Reference to VirtualLab PYTHONPATH found in .bashrc"
    echo "Therefore, not adding again."
  else
    echo "Adding $VL_DIR to PYTHONPATH"
    sudo -u ${SUDO_USER:-$USER} echo 'export PYTHONPATH=$PYTHONPATH'$VL_DIR''  >> ~/.bashrc
    export PYTHONPATH=$PYTHONPATH$VL_DIR
  fi
elif [ "$CONDA_INST" == "y" ]; then
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
    echo "There has been a problem installing Conda"
    echo "Check error messages, try to rectify then rerun this script"
  fi
  #conda --version

  # Install conda packages
  echo "Creating Conda env $CONDAENV"
  conda create -n $CONDAENV python -y
  conda activate $CONDAENV
  conda config --append channels conda-forge
  conda install -y numpy scipy matplotlib pillow h5py iapws

  # Install python and required packages
  sudo apt install -y python3-pip
  sudo -u ${SUDO_USER:-$USER} pip3 install fpdf
  #sudo -u ${SUDO_USER:-$USER} pip3 install fpdf2
  echo "Finished creating Conda env $CONDAENV"

  # Add $VL_DIR to $PYTHONPATH in Conda env and current shell
  PYV=`python -V`
  PYV2=${PYV#* }
  PYV=${PYV2%.*}
  PATH_FILE=$HOME/anaconda3/envs/$CONDAENV/lib/python$PYV/site-packages/VirtualLab.pth
  if test -f "$PATH_FILE"; then
    echo "VirtualLab PYTHONPATH found in Conda env"
  else
    echo "Adding $VL_DIR to PYTHONPATH in Conda env"
    sudo -u ${SUDO_USER:-$USER} echo $VL_DIR >> $PATH_FILE
    export PYTHONPATH=$PYTHONPATH$VL_DIR
  fi
else
  echo "Error: You shouldn't have reached this part of the script..."
  echo "How did you manage that?!?"
  exit
fi

