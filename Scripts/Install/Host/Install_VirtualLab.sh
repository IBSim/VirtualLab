#!/bin/bash

USER_HOME=$(eval echo ~"${SUDO_USER}")
if [ -f "$USER_HOME"/.VLprofile ]; then source "$USER_HOME"/.VLprofile; fi

usage() {
  echo
  echo "Usage:"
  echo " $0 [-B {m/d}]"
  echo
  echo "A script to install VirtualLab and its dependencies with default settings."
  echo
  echo "Options:"
  echo "   '-B m' Install the main (stable) version of VirtualLab"
  echo "   '-B d' Install the dev (developmental) version of VirtualLab"
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

### Default VirtualLab branch if no flag.
BRANCH=master

################################################################################
#                    Parse CMD Arguments
################################################################################

while getopts "B:yh" options; do
  case "${options}" in
    B)
      BRANCH=${OPTARG}
      if [ "$BRANCH" != "master" ] || [ "$BRANCH" != "dev" ] ; then
        echo "############## Warning #############"
        echo
        echo "VirtualLab not installed from master or dev branch. Errors may occur"
        echo
        echo "####################################"
      else
        echo " - VirtualLab will be installed from branch $BRANCH."
      fi
      ;;
    y)  ### Skip install confirmation dialogue.
      SKIP=y
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

### Double check with user that they're happy to continue.
### This is skippable with -y flag.
if [[ ! "$SKIP" =~ "y" ]]; then
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
if hash VirtualLab 2>/dev/null; then
  ### If exists, do nothing
  echo "VirtualLab exists in PATH"
  echo "Skipping VirtualLab installation"
else

  echo "VirtualLab not found on the system, continuing with installation."
  ### Standard update
  sudo apt update -y
  sudo apt upgrade -y
  sudo apt install -y build-essential

  cd $USER_HOME
  #echo $USER_HOME
  #pwd
  
#: <<'END'

  ### Install git
  fname=Install_git.sh
  url="https://gitlab.com/ibsim/virtuallab/-/raw/"$BRANCH"/Scripts/Install/Host/"$fname
  #echo $url
  #url="https://gitlab.com/ibsim/virtuallab_bin/-/raw/"$BRANCH"/"$fname
  wget $url
  chmod 755 $fname
  ./$fname
  rm $fname
  
  ### Install Apptainer
  fname=Install_Apptainer-bin.sh
  url="https://gitlab.com/ibsim/virtuallab/-/raw/"$BRANCH"/Scripts/Install/Host/"$fname
  wget $url
  chmod 755 $fname
  ./$fname
  rm $fname
  
  ### Download VirtualLab repo and configure it on the system
  # all arguments are passed to 
  fname=Install_VL.sh
  url="https://gitlab.com/ibsim/virtuallab/-/raw/"$BRANCH"/Scripts/Install/Host/"$fname
  wget $url
  chmod 755 $fname
  sudo -u ${SUDO_USER:-$USER} ./$fname -B $BRANCH #$@
  rm $fname
  
  ### Test to check if installation worked
  source .VLprofile
  if hash VirtualLab 2>/dev/null; then
    ### If exists
    echo "VirtualLab has been installed"
  else
    ### VirtualLab still not installed
    echo "There has been a problem installing VirtualLab"
    echo "Check error messages, try to rectify then rerun this script"
    exit
    
  fi
#END
fi
