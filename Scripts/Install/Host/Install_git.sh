#!/bin/bash

set -e
USER_HOME=$(eval echo ~${SUDO_USER})
#if [ -f $USER_HOME/.VLprofile ]; then source $USER_HOME/.VLprofile; fi

#########################
### This script is used to install git and its dependencies.
### It first attempts to detect whether it is already installed.
#########################

### Standard update
sudo apt update -y
sudo apt upgrade -y
sudo apt install -y build-essential

### Test to check if git already exists in current shell's PATH
if hash git 2>/dev/null; then
  ### If exists, do nothing
  echo "git exists in PATH"
  echo "Skipping git installation"
else

  echo
  echo "Installing git"
  echo "~~~~~~~~~~~~~~"
  echo

  sudo apt install git -y

  ### Test to check if installation worked
  if hash git 2>/dev/null; then
    ### If exists
    echo "git has been installed"
  else
    ### git still not installed
    echo "There has been a problem installing git"
    echo "Check error messages, try to rectify then rerun this script"
    exit
  fi
fi
