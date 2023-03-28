#!/bin/bash

USER_HOME=$(eval echo ~"${SUDO_USER}")

usage() {
  echo
  echo "Usage:"
  echo " $0 [-B {m/d}]"
  echo
  echo "A script to install VirtualLab with default settings."
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
        BRANCH=main
        B_UC=M
        echo " - VirtualLab will be installed from the main branch."
      elif [ "$B" == "d" ]; then
        BRANCH=dev
        B_UC=D
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

### Install VirtualLab
cd $USER_HOME
fname=Install_VirtualLab
url="https://gitlab.com/ibsim/virtuallab_bin/-/raw/"$BRANCH"/"$fname
#wget https://gitlab.com/ibsim/virtuallab_bin/-/raw/$BRANCH/Install_VirtualLab
wget $url

chmod 755 $fname
./$fname -y -B $B_UC #&> ~/Install_VL.log
rm $fname
