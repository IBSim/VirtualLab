#!/bin/bash
set -e
USER_HOME=$(eval echo ~${SUDO_USER})
if [ -f $USER_HOME/.VLprofile ]; then source $USER_HOME/.VLprofile; fi

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $THIS_DIR

source "$VL_DIR/VLconfig.py" # Enables this script to be run seperately
#source ../../VLconfig.py # probably an easier way to do it but will keep it consistent

### To get from VLconfig.py
#ERMES_DIR="/opt/ERMES"
#ERMES_VER="ERMES-CPlas-v12.5"

### Test to check if ERMES already exists in current shell's PATH
if hash ERMESv12.5 2>/dev/null; then
  ### If exists, do nothing
  echo "ERMES exists in PATH"
  echo "Skipping ERMES installation"
else
  ### Do more checks
  echo "ERMES does not exist in this shell's environment PATH"
  ### Search for reference to ERMES in ~/.VLprofile
  STRING_TMP="$ERMES_DIR/$ERMES_VER"
  if [[ $(grep -q "$STRING_TMP" $USER_HOME/.VLprofile | grep -F -v "#") ]]; then
    echo "Reference to ERMES PATH found in .VLprofile"
    echo "Assuming ERMES is installed"
    echo "Skipping ERMES installation"
    ### Execute output from grep to try and add to shell's PATH
    source <(grep "STRING_TMP" $USER_HOME/.VLprofile)
  else
    ### Download ERMES
    echo "Downloading and installing ERMES in $ERMES_DIR"
    wget https://ibsim.co.uk/VirtualLab/downloads/$ERMES_VER.zip
    sudo unzip $ERMES_VER.zip -d $ERMES_DIR

    ### Make exe
    sudo chmod 755 $ERMES_DIR/$ERMES_VER/ERMESv12.5

    ### Add to PATH
    echo "Adding ERMES to PATH"
    sudo -u ${SUDO_USER:-$USER} echo 'if [[ ! $PATH =~ "'$ERMES_DIR'/'$ERMES_VER'" ]]; then' >> $USER_HOME/.VLprofile
    sudo -u ${SUDO_USER:-$USER} echo '  export PATH="'$ERMES_DIR'/'$ERMES_VER':$PATH"'  >> $USER_HOME/.VLprofile
    sudo -u ${SUDO_USER:-$USER} echo 'fi'  >> $USER_HOME/.VLprofile
    export PATH="$ERMES_DIR"/"$ERMES_VER:$PATH"

    ### ~/.bashrc doesn't get read by subshells in ubuntu.
    ### Workaround: store additions to env PATH in ~/.VLprofile & source in bashrc.
    STRING_TMP="if [ -f ~/.VLprofile ]; then source ~/.VLprofile; fi"
    if [[ ! $(grep -F "$STRING_TMP" $USER_HOME/.bashrc | grep -F -v "#$STRING") ]]; then
      echo $STRING_TMP >> $USER_HOME/.bashrc
    fi

    ### Test to check if adding to path worked
    if hash ERMESv12.5 2>/dev/null; then
      ### If exists
      echo "ERMES now exists in PATH"
      ### ADD TEST HERE TO CHECK WORKING AS EXPECTED
      ### If all working rm download files
    else
      ### ERMES still not installed
      echo "There has been a problem installing ERMES"
      echo "Check error messages, try to rectify then rerun this script"
    fi
  fi
fi
