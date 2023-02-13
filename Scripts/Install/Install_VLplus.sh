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
BRANCH=main

################################################################################
#                    Parse CMD Arguments
################################################################################

while getopts "B:yh" options; do
  case "${options}" in
    B)
      B=${OPTARG}
      if [ "$B" == "m" ]; then
        BRANCH=master
        echo " - VirtualLab will be installed from the main branch."
      elif [ "$B" == "d" ]; then
        BRANCH=dev
        echo " - VirtualLab will be installed from the dev branch."
      else
        echo "Error: Invalid option argument $BRANCH" >&2
        exit_abnormal
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
    *)  ### If unknown (any other) option:
      echo "Error: Invalid option -$OPTARG" >&2
      exit_abnormal
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

#########################
### This script is used to install VirtualLab and its dependencies.
### It first attempts to detect whether it is already installed.
#########################

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
  url="https://gitlab.com/ibsim/virtuallab/-/raw/"$BRANCH"/Scripts/Install/"$fname
  #echo $url
  #url="https://gitlab.com/ibsim/virtuallab_bin/-/raw/"$BRANCH"/"$fname
  wget $url
  chmod 755 $fname
  ./$fname
  rm $fname
  
  ### Install Apptainer
  fname=Install_Apptainer-bin.sh
  url="https://gitlab.com/ibsim/virtuallab/-/raw/"$BRANCH"/Scripts/Install/"$fname
  wget $url
  chmod 755 $fname
  ./$fname
  rm $fname
  
  ### Install VirtualLab
  fname=Install_VL_Container.sh
  url="https://gitlab.com/ibsim/virtuallab/-/raw/"$BRANCH"/Scripts/Install/"$fname
  wget $url
  chmod 755 $fname
  sudo -u ${SUDO_USER:-$USER} ./$fname -B $B -y
  rm $fname
  
  ### Test to check if installation worked
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
