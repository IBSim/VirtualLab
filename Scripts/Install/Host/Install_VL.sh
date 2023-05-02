#!/bin/bash

# Script for downloading the VirtualLab repo and configuriging its use on the host system. 

USER_HOME=$(eval echo ~"${SUDO_USER}")

usage() {
  echo
  echo "Usage:"
  echo " $0 [-B {name} -V {name}]"
  echo
  echo "A script to install VirtualLab with default settings."
  echo
  echo "Options:"
  echo "   '-B {branch name}' Branch from VirtualLab repo (default master)"  
  echo "   '-E {branch name}' Branch from binary repo (defaul main)"
  echo "   '-y' Skip install confirmation dialogue."
}

exit_abnormal() {
  usage
  exit 1
}

### Default VirtualLab branch if no flag.
BRANCH_VL=master
BRANCH_bin=main

################################################################################
#                    Parse CMD Arguments
################################################################################

while getopts "B:E:yh" options; do
  case "${options}" in
    B)
      # VirtualLab branch
      BRANCH_VL=${OPTARG}
      if [ "$BRANCH_VL" == "master" ]; then
        BRANCH_bin=main
      else;
        BRANCH_bin=$BRANCH_VL
      fi
      ;;
    E)
      # VirtualLab executable branch
      BRANCH_VL=${OPTARG}
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

#echo " - VirtualLab will be installed from the main branch."

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

### Install VirtualLab using installation binary
cd $USER_HOME
fname=Install_VirtualLab
url="https://gitlab.com/ibsim/virtuallab_bin/-/raw/"$BRANCH_bin"/"$fname
wget $url

chmod 755 $fname
./$fname -y -B $BRANCH_VL #&> ~/Install_VL.log
rm $fname

