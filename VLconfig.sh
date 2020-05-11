#!/bin/bash

#########################
### START OF DEFAULTS ###
### These are the default values.
### This script will detect whether you already have VirtualLab installed,
### if found it will replace default values with current values.
### Change the values below if you would like to use non default values.
VL_DIR_NAME_DEFAULT="VirtualLab"
VL_DIR_DEFAULT="$HOME/$VL_DIR_NAME_DEFAULT"
CONDA_VER_DEFAULT="Anaconda3-2020.02-Linux-x86_64.sh"
SALOME_DIR_DEFAULT="/opt/SalomeMeca"
SALOME_VER_DEFAULT="salome_meca-2019.0.3-1-universal"
SALOME_BIN_DEFAULT="V2019.0.3_universal"
ASTER_DIR_DEFAULT="/opt/SalomeMeca/V2019.0.3_universal/tools/Code_aster_frontend-20190/bin/as_run"
InputDir_DEFAULT="$VL_DIR_DEFAULT/Input"
MaterialsDir_DEFAULT="$VL_DIR_DEFAULT/Materials"
RunFilesDir_DEFAULT="$VL_DIR_DEFAULT/RunFiles"
OutputDir_DEFAULT="$VL_DIR_DEFAULT/Output"
TEMP_DIR_DEFAULT="/tmp"
### END OF DEFAULTS ###
#######################

### This is a list of the variables to be exported from this file.
### Add any additional variables to this list to be sourced by SetupConfig.
var=(
  VL_DIR_NAME
  VL_DIR
  CONDA_VER
  SALOME_DIR
  SALOME_VER
  SALOME_BIN
  ASTER_DIR
  InputDir
  MaterialsDir
  RunFilesDir
  OutputDir
  TEMP_DIR
)

### This list of config variables is being ready by a bash script.
###  - Variables included here are default values. Others are untested.
###  - Test use of variables for compatibility with bash.
###  - Do not leave spaces next to equal signs.

### PATH to VirtualLab directory.
### This is to enable running VirtualLab from locations other than the source 
### top directory. If left commented it is only possible to run from the 
### location where VirtualLab is installed.
###
### Find hidden file with long random string in VirtualLab's top direcory.

# Wedi cuddio y rhan yma i neud y proses mwy cloi

#VL_find=$(find / -iname ".1EU3DDeS1Zu57zby" 2>/dev/null -printf '%h\n')
#if test -z $VL_find; then
#  # VirtualLab doesn't already exist on the system.
#  VL_DIR_NAME=$VL_DIR_NAME_DEFAULT
#  VL_DIR=$VL_DIR_DEFAULT
#else
#  # VirtualLab found to already exist on the system.
#  VL_DIR_NAME="$(basename "$VL_find")"
#  VL_DIR="$(dirname "$VL_find")/$VL_DIR_NAME"
#fi

VL_DIR_NAME="VirtualLab"
VL_DIR="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
#VL_DIR="/home/rhydian/Documents/Scripts/Simulation/VirtualLab"

### Version of conda to download and install with
### wget https://repo.anaconda.com/archive/"$CONDA_VER"
CONDA_VER=$CONDA_VER_DEFAULT

### Salome/Code_Aster download, install and config variables.
### Salome version number in download filename
SALOME_VER=$SALOME_VER_DEFAULT

### Test to check if salome already exists in current shell's PATH.
prefix='appli_'
if hash salome 2>/dev/null; then
  # If exists, find PATHs
  echo "Salome exists in PATH, using values based on that."
  SALOME_PATH=$(which salome); #echo $SALOME_PATH
  SALOME_TMP="$(dirname "$SALOME_PATH")"; #echo $SALOME_TMP
  SALOME_BIN="$(basename "$SALOME_TMP")"; #echo $SALOME_BIN
  SALOME_DIR="$(dirname "$SALOME_TMP")"; #echo $SALOME_DIR
  SALOME_BIN=${SALOME_BIN#"$prefix"}; #echo "${SALOME_BIN}"
else
  # Do more checks, search in /opt.
  echo "Salome does not exist in this shell's environment PATH."
  if test $(find /opt -iname "SalomeMeca" 2>/dev/null); then
    # Do something
    echo "Salome found in /opt, using values based on that."
    SALOME_DIR='/opt/SalomeMeca'; #echo $SALOME_DIR
    SALOME_TMP=$(find $SALOME_DIR -maxdepth 1 -name $prefix* 2>/dev/null); #echo $SALOME_TMP
    SALOME_TMP="$(basename "$SALOME_TMP")"; #echo $SALOME_TMP
    SALOME_BIN=${SALOME_TMP#"$prefix"}; #echo "${SALOME_BIN}"
  else
    ### Salome not found on system.
    echo "Salome not in path or in /opt, using default values."
    ### Salome installation location
    SALOME_DIR=$SALOME_DIR_DEFAULT
    ### Salome version number in unpacked directory
    SALOME_BIN=$SALOME_BIN_DEFAULT
  fi
fi

### Code_Aster installation location
ASTER_DIR=$ASTER_DIR_DEFAULT

### PATH to various directories required as in/out for VirtualLab.
### If left commented default behaviour is to locate in $VL_DIR

InputDir="$VL_DIR/Input"
MaterialsDir="$VL_DIR/Materials"
RunFilesDir="$VL_DIR/RunFiles"
OutputDir="$VL_DIR/Output"
TEMP_DIR=$TEMP_DIR_DEFAULT
