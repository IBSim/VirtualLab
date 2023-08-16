#!/bin/bash
set -e
USER_HOME=$(eval echo ~${SUDO_USER})

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $THIS_DIR

ERMES_DIR="/opt/ERMES"
ERMES_VER="ERMES-CPlas-v12.5"

### Test to check if ERMES already exists in current shell's PATH
if hash ERMESv12.5 2>/dev/null; then
  ### If exists, do nothing
  echo "ERMES exists in PATH"
  echo "Skipping ERMES installation"
else
  ### Download ERMES
  echo "Downloading and installing ERMES in $ERMES_DIR"
  wget https://ibsim.co.uk/VirtualLab/downloads/$ERMES_VER.zip
  sudo unzip $ERMES_VER.zip -d $ERMES_DIR

  ### Make exe
  sudo chmod 755 $ERMES_DIR/$ERMES_VER/ERMESv12.5

  ### Add to PATH
  echo "Adding ERMES to PATH"
  sudo -u ${SUDO_USER:-$USER} echo '  export PATH="'$ERMES_DIR'/'$ERMES_VER':$PATH"'  >> $USER_HOME/.bashrc
  export PATH="$ERMES_DIR"/"$ERMES_VER:$PATH"

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
