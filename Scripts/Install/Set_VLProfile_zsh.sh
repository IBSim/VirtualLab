#!/bin/bash
###########################################################
# script to set setup .VLprofile for MacOs and and Zsh based
# Linux installs. VLprofile is a text file that gets sourced 
# by .zshrc on startup. this sets some enviroment variables 
# to add VirtualLab to the system path and tell VirtualLab 
# where the code has been installed.
#
# Takes two arguments the first is the abs path to the 
# virtualLab dir and the second is path to the users home 
# directory.
##########################################################
set -e
VL_DIR=$1
USER_HOME=$2
### Check if VirtualLab is in PATH
if [[ $PATH =~ $VL_DIR ]]; then
  echo "VirtualLab is already in PATH."
else
  ### If not, add VirtualLab to PATH
  echo "Adding VirtualLab to PATH."
  # Add VL_DIR to VLProfile so that different parts of install can be run seperately
  echo 'export VL_DIR="'$VL_DIR'"' >> $USER_HOME/.VLprofile

  echo 'if [[ ! $PATH =~ "'$VL_DIR'" ]]; then' >> $USER_HOME/.VLprofile

  echo '  export PATH="'$VL_DIR'/bin:$PATH"'  >> $USER_HOME/.VLprofile
  echo 'fi'  >> $USER_HOME/.VLprofile

  export PATH="$VL_DIR/bin:$PATH"
fi

### ~/.bashrc doesn't get read by subshells in ubuntu.
### Workaround: store additions to env PATH in ~/.VLprofile & source in bashrc.
STRING_TMP="if [ -f ~/.VLprofile ]; then source ~/.VLprofile; fi"
if [[ ! $(grep -F "$STRING_TMP" $USER_HOME/.bashrc | grep -F -v "#$STRING") ]]; then
  echo '' >> $USER_HOME/.zshrc
  echo '# Read in environment for VirtualLab' >> $USER_HOME/.zshrc
  echo $STRING_TMP >> $USER_HOME/.zshrc
fi
set +e