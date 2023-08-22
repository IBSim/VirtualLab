#!/bin/bash

USER_HOME=$(eval echo ~"${SUDO_USER}")
if [ -f "$USER_HOME"/.VLprofile ]; then source "$USER_HOME"/.VLprofile; fi

usage() {
  echo
  echo "Usage:"
  echo " $0 [-B {name}] [-I {python/conda/binary}] "
  echo
  echo "A script to install VirtualLab with default settings."
  echo
  echo "   '-B {Branch_name}' Install VirtualLab from Branch_name (default is master)"
  echo "   '-g {y/n} flag for whether or not to install git (default is y) "
  echo "   '-a {y/n} flag for whether or not to install apptainer (default is y) "  
  echo "   '-I {python/conda/binary/N}' How VirtualLab is installed, either using standard python (default), conda or pre-built binaries."
  echo "                                N will not install any new packages, giving the user more flexibility."
  echo "   '-d {directory}' Path to custom directory in which to install VirtuaLab"
  echo "   '-y' Skip install confirmation dialogue."
}

exit_abnormal() {
  usage
  exit 1
}

if [[ $EUID -ne 0 ]]; then
   echo "This installation script must be run as root"
   echo 'Re-run with "sudo ./Install_VLplus.sh {options}".'
   exit_abnormal
fi

### Default values for parsed arguments
BRANCH=master
VL_BINARY='python'
GIT=y
APPTAINER=y
SKIP=""
INST_DIR=""


################################################################################
#                    Parse CMD Arguments
################################################################################

while getopts "B:g:a:I:d:yh" options; do
  case "${options}" in
    B)
      BRANCH="${OPTARG}"
      ;;
    g)
      GIT="${OPTARG}"
      ;;
    a)
      APPTAINER="${OPTARG}"
      ;;        
    I) # how installation and binaries are created
      VL_BINARY="${OPTARG}"
      if ! ([ "$VL_BINARY" = "python" ] || [ "$VL_BINARY" = "conda" ] || [ "$VL_BINARY" = "binary" ]) ; then
        echo "Error: Invalid argument $VL_BINARY for option I" >&2
        exit_abnormal      
      fi
      ;;
    d) 
      INST_DIR="-d ${OPTARG}"
      ;;
    y)  ### Skip install confirmation dialogue.
      SKIP=-y
      ;;
    h)  ### display Help
      exit_abnormal
      ;;
    :)  ### If expected argument omitted:
      echo "Error: Option -${OPTARG} requires an argument."
      exit_abnormal
      ;;
    *)
      ;;
  esac
done

### Check that no additional args were given that weren't caught.
shift $(($OPTIND - 1))
if [[ $@ ]]; then
  echo
  echo "Error: Too many arguments were entered, please check."
  echo "Remaining arguments: \"$@\"."
  exit_abnormal
fi


# provide information about install 
echo
echo "Installing VirtualLab from branch $BRANCH using $VL_BINARY."
echo

if ! ([ "$BRANCH" = "master" ] || [ "$BRANCH" = "dev" ]) ; then
  echo "############## Warning #############"
  echo
  echo "Errors may occur when installing from branches other than master or dev"
  echo
  echo "####################################"
fi

### Double check with user that they're happy to continue.
### This is skippable with -y flag.
if [[ ! "$SKIP" =~ "y" ]]; then
  echo
  read -r -p "Are you sure? [y/n] " response
  if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo "Make it so!"
  else
    echo "Exiting VirtualLab installation/configuration."
    exit
  fi
fi

########################
## This script is used to install VirtualLab and its dependencies.
## It first attempts to detect whether it is already installed.
########################

### Test to check if VirtualLab already exists in current shell's PATH
# if hash VirtualLab 2>/dev/null; then
#   ### If exists, do nothing
#   echo "VirtualLab exists in PATH"
#   echo "Skipping VirtualLab installation"


### Standard update
sudo apt update -y
sudo apt upgrade -y
sudo apt install -y build-essential

cd $USER_HOME
INST_PATH="https://gitlab.com/ibsim/virtuallab/-/raw/"$BRANCH"/Scripts/Install/Host"

if [ "$GIT" == "y" ]; then
  ### Install git

  fname=Install_git.sh
  wget "${INST_PATH}"/"${fname}" 
  chmod 755 $fname
  ./$fname
  rm $fname
fi

if [ "$APPTAINER" == "y" ]; then
  ### Install Apptainer
  fname=Install_Apptainer-bin.sh
  wget "${INST_PATH}"/"${fname}" 
  chmod 755 $fname
  ./$fname
  rm $fname
fi

### Download VirtualLab repo and configure it on the system
fname=Install_VirtualLab.sh
wget "${INST_PATH}"/"${fname}" 
chmod 755 $fname
sudo -u ${SUDO_USER:-$USER} ./$fname -B $BRANCH -I $VL_BINARY $INST_DIR $SKIP -Z
rm $fname


