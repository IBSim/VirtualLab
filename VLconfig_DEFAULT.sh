#!/bin/bash
USER_HOME=$(eval echo ~${SUDO_USER})
if [ -f $USER_HOME/.VLprofile ]; then source $USER_HOME/.VLprofile; fi

#SALOME_INST="n"
### This caused Install_VirtualLab to skip Salome install.
### Being commented out might cause issues with SetupConfig if run without install.

#########################
### START OF DEFAULTS ###
### These are the default config values for installation and operation of VirtualLab.
### This script will detect whether you already have VirtualLab installed.
### If found, it will replace default values with current values.
### Change the values below if you would like to use non default values.
### Paths will be relative to $VL_DIR (i.e. the VirtualLab installation directory).
### You must then run SetupConfig.sh to create VLconfig.py as input for VirtualLab.
#########################

### Config values for VirtualLab installation.
VL_DIR_DEFAULT="$HOME/VirtualLab"
CONDA_VER_DEFAULT="Anaconda3-2020.02-Linux-x86_64.sh"
SALOME_DIR_DEFAULT="/opt/SalomeMeca"
SALOME_VER_DEFAULT="salome_meca-2019.0.3-1-universal"
SALOME_BIN_DEFAULT="V2019.0.3_universal"

#########################

### Config values for VirtualLab operation.
ASTER_DIR_DEFAULT="/opt/SalomeMeca/V2019.0.3_universal/tools/Code_aster_frontend-20190/bin/as_run"
InputDir_DEFAULT="$VL_DIR_DEFAULT/Input"
MaterialsDir_DEFAULT="$VL_DIR_DEFAULT/Materials"
RunFilesDir_DEFAULT="$VL_DIR_DEFAULT/RunFiles"
OutputDir_DEFAULT="$VL_DIR_DEFAULT/Output"
TEMP_DIR_DEFAULT="/tmp"

### END OF DEFAULTS ###
#######################

### DO NOT CHANGE ANYTHING BENEATH THIS LINE ###

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

### Verbose {ON/OFF}
v="OFF"

### Replace '~' with '$HOME' if present
for i in ${!var[@]}; do
  var_def_val=${var[i]}_DEFAULT
  STRING_TMP=${!var_def_val}
  STRING_TMP=${STRING_TMP/'~'/'$HOME'}
  var_def=${var[$i]}_DEFAULT
  eval "${var_def}=$STRING_TMP"
done

### Locate PATH to VirtualLab directory to enable running VirtualLab from
### locations other than the source top directory.
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
VL_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"

### Checks whether $VL_DIR_DEFAULT is used in any variables.
### If so and $VL_DIR is different, will replace string.
if [[ ! "$VL_DIR_DEFAULT" =~ "$VL_DIR" ]]; then
  for i in ${!var[@]}; do
    var_def_val=${var[i]}_DEFAULT
    STRING_TMP=${!var_def_val}
    var_def=${var[$i]}_DEFAULT
    if [[ $STRING_TMP == *"$VL_DIR_DEFAULT"* ]] && [[ ! $var_def == "VL_DIR_DEFAULT" ]]; then
      STRING_TMP="${STRING_TMP/$VL_DIR_DEFAULT/$VL_DIR}"
      eval "${var_def}=$STRING_TMP"
    fi
  done
fi

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
  if [[ $v == "ON" ]]; then echo "Salome exists in PATH, using values based on that."; fi
  SALOME_PATH=$(which salome); #echo $SALOME_PATH
  SALOME_TMP="$(dirname "$SALOME_PATH")"; #echo $SALOME_TMP
  SALOME_BIN="$(basename "$SALOME_TMP")"; #echo $SALOME_BIN
  SALOME_DIR="$(dirname "$SALOME_TMP")"; #echo $SALOME_DIR
  SALOME_BIN=${SALOME_BIN#"$prefix"}; #echo "${SALOME_BIN}"
else
  ### Do more checks, search in /opt.
  if [[ $v == "ON" ]]; then echo "Salome does not exist in this shell's environment PATH."; fi
  if test $(find /opt -iname "SalomeMeca" 2>/dev/null); then
    ### Set PATHs based on that.
    if [[ $v == "ON" ]]; then echo "Salome found in /opt, using values based on that."; fi
    SALOME_DIR='/opt/SalomeMeca'; #echo $SALOME_DIR
    SALOME_TMP=$(find $SALOME_DIR -maxdepth 1 -name $prefix* 2>/dev/null); #echo $SALOME_TMP
    SALOME_TMP="$(basename "$SALOME_TMP")"; #echo $SALOME_TMP
    SALOME_BIN=${SALOME_TMP#"$prefix"}; #echo "${SALOME_BIN}"
  else
    ### Salome not found on system.
    if [[ "$SALOME_INST" =~ 'y' ]] && [[ ! -z "$SALOME_DIR" ]]; then
      if [[ $v == "ON" ]]; then echo "Salome not in env path or in /opt, it will be installed in '$SALOME_DIR'."; fi
    else
      if [[ $v == "ON" ]]; then echo "Salome not in env path or in /opt, using default values."; fi
      ### Salome installation location
      SALOME_DIR=$(readlink -m $SALOME_DIR_DEFAULT)
    fi
    ### Salome version number in unpacked directory
    SALOME_BIN=$SALOME_BIN_DEFAULT
  fi
fi

### Code_Aster installation location
### Currently, it could be possible to combine if and elif conditions below.
### Not doing so, in case they need to be separate in future.
if [[ "$SALOME_INST" =~ 'y' ]] && [[ "$SALOME_DIR" != $(readlink -m $SALOME_DIR_DEFAULT) ]]; then
  ASTER_DIR=$SALOME_DIR$ASTER_SUBDIR
  if [[ $v == "ON" ]]; then
    echo "This condition should be: no salome installed, SALOME_DIR set with flag"
    echo "ASTER_DIR should be set from Install_VirtualLab.sh"
    echo "ASTER_DIR = "$ASTER_DIR
  fi
elif [[ "$SALOME_INST" =~ 'y' ]] && [[ "$SALOME_DIR" == $(readlink -m $SALOME_DIR_DEFAULT) ]]; then
  #ASTER_DIR=$(readlink -m $ASTER_DIR_DEFAULT)
  ASTER_DIR=$SALOME_DIR$ASTER_SUBDIR
  if [[ $v == "ON" ]]; then
    echo "This condition should be: no salome installed, SALOME_DIR NOT set with flag"
    echo "ASTER_DIR should be set from Install_VirtualLab.sh, but using /opt/Salome as SALOME_DIR"
    echo "ASTER_DIR = "$ASTER_DIR
  fi
else
  if [[ $v == "ON" ]]; then echo "This condition should be: salome already installed, ASTER_DIR read from above"; fi
  ASTER_DIR=$(readlink -m $ASTER_DIR_DEFAULT)
fi

### PATH to various directories required as in/out for VirtualLab.
### Default behaviour is to locate in $VL_DIR.
InputDir=$(readlink -m $InputDir_DEFAULT)
MaterialsDir=$(readlink -m $MaterialsDir_DEFAULT)
RunFilesDir=$(readlink -m $RunFilesDir_DEFAULT)
OutputDir=$(readlink -m $OutputDir_DEFAULT)
TEMP_DIR=$(readlink -m $TEMP_DIR_DEFAULT)

