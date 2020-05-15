#!/bin/bash
if [ -f ~/.profile ]; then source ~/.profile; fi

#########################
### START OF DEFAULTS ###
### These are the default values.
### This script will detect whether you already have VirtualLab installed,
### if found it will replace default values with current values.
### Change the values below if you would like to use non default values.
VL_DIR_DEFAULT="$HOME/VirtualLab"
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
echo '$VL_DIR_DEFAULT/Input = '"$VL_DIR_DEFAULT/Input"
echo "InputDir_DEFAULT = "$InputDir_DEFAULT
### This is a list of the variables to be exported from this file.
### Add any additional variables to this list to be sourced by SetupConfig.
var=(
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

### Locate PATH to VirtualLab directory to enable running VirtualLab from
### locations other than the source top directory.
#echo "VLconf $VL_DIR"
VL_DIR="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

### Version of conda to download and install with
### wget https://repo.anaconda.com/archive/"$CONDA_VER"
CONDA_VER=$CONDA_VER_DEFAULT

### Salome/Code_Aster download, install and config variables.
### Salome version number in download filename
SALOME_VER=$SALOME_VER_DEFAULT

### Test to check if salome already exists in current shell's PATH.
prefix='appli_'
if hash salome 2>/dev/null; then
  ### If exists, find PATHs
  echo "Salome exists in PATH, using values based on that."
  SALOME_PATH=$(which salome); #echo $SALOME_PATH
  SALOME_TMP="$(dirname "$SALOME_PATH")"; #echo $SALOME_TMP
  SALOME_BIN="$(basename "$SALOME_TMP")"; #echo $SALOME_BIN
  SALOME_DIR="$(dirname "$SALOME_TMP")"; #echo $SALOME_DIR
  SALOME_BIN=${SALOME_BIN#"$prefix"}; #echo "${SALOME_BIN}"
else
  ### Do more checks, search in /opt.
  echo "Salome does not exist in this shell's environment PATH."
  if test $(find /opt -iname "SalomeMeca" 2>/dev/null); then
    ### Set PATHs based on that.
    echo "Salome found in /opt, using values based on that."
    SALOME_DIR='/opt/SalomeMeca'; #echo $SALOME_DIR
    SALOME_TMP=$(find $SALOME_DIR -maxdepth 1 -name $prefix* 2>/dev/null); #echo $SALOME_TMP
    SALOME_TMP="$(basename "$SALOME_TMP")"; #echo $SALOME_TMP
    SALOME_BIN=${SALOME_TMP#"$prefix"}; #echo "${SALOME_BIN}"
  else
    ### Salome not found on system.
    if [[ "$SALOME_INST" =~ 'y' ]] & [[ ! -z "$SALOME_DIR" ]]; then
      echo "Salome not in env path or in /opt, it will be installed in '$SALOME_DIR'."
    else
      echo "Salome not in env path or in /opt, using default values."
      ### Salome installation location
      SALOME_DIR=$(readlink -m $SALOME_DIR_DEFAULT)
    fi
    ### Salome version number in unpacked directory
    SALOME_BIN=$SALOME_BIN_DEFAULT
  fi
fi
### Code_Aster installation location
ASTER_DIR=$(readlink -m $ASTER_DIR_DEFAULT)

### PATH to various directories required as in/out for VirtualLab.
### Default behaviour is to locate in $VL_DIR.
#echo '$InputDir_DEFAULT = '$InputDir_DEFAULT
#echo '$InputDir = '$InputDir

#STRING_TMP=$InputDir_DEFAULT
#STRING_TMP=${STRING_TMP/'$VL_DIR_DEFAULT/'/''}
#echo '$STRING_TMP = '$STRING_TMP
#InputDir=$(readlink -m $STRING_TMP)

#InputDir=$(readlink -m $InputDir_DEFAULT)
#echo "dirname = "$(dirname "$InputDir")
#echo "dirnamex2 = "$(dirname "$(dirname "$InputDir")")
#echo "basename = "$(basename "$InputDir")
#echo "path = "$(dirname "$(dirname "$InputDir")")"/"$(basename "$InputDir")

InputDir=$(readlink -m $InputDir_DEFAULT)
MaterialsDir=$(readlink -m $MaterialsDir_DEFAULT)
RunFilesDir=$(readlink -m $RunFilesDir_DEFAULT)
OutputDir=$(readlink -m $OutputDir_DEFAULT)
TEMP_DIR=$(readlink -m $TEMP_DIR_DEFAULT)

#echo '$InputDir_DEFAULT = '$InputDir_DEFAULT
#echo '$InputDir = '$InputDir

