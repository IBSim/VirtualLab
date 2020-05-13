#!/bin/bash

VL_DIR_NAME="VirtualLab"
VL_DIR="$HOME/$VL_DIR_NAME"

usage() {
  echo
  echo "Usage:"
  echo " $0 [ -P {y/c/n} ] [ -S {y/n} ]"
  echo
  echo "A script to install VirtualLab and its dependencies."
  echo
  echo "Options:"
  echo "   '-P y' Install python using local installation"
  echo "   '-P c' Install python using conda envrionment"
  echo "   '-P n' Do not install python"
  echo "   '-S y' Install Salome-Meca"
  echo "   '-S n' Do not install Salome-Meca"
  echo
  echo "Default behaviour is to not install -P or -S."
}
exit_abnormal() {
  usage
  exit 1
}
while getopts ":P:S:h" options; do 
  case "${options}" in
    P)
      PYTHON_INST=${OPTARG}
      if [ "$PYTHON_INST" == "y" ]; then
        echo "Python will be installed/updated and configured as part of VirtualLab install."
      elif [ "$PYTHON_INST" == "c" ]; then
        echo "Conda will be installed/updated and configured as part of VirtualLab install."
      elif [ "$PYTHON_INST" == "n" ]; then
        echo "Python will not be installed or configured during setup, please do this manually."
      else
        echo "Error: Invalid option argument $PYTHON_INST" >&2
        exit_abnormal
      fi
      ;;
    S)
      SALOME_INST=${OPTARG}
      if [ "$SALOME_INST" == "y" ]; then
        echo "Salome-Meca will be installed and configured as part of VirtualLab install."
      elif [ "$SALOME_INST" == "n" ]; then
        echo "Salome-Meca will not be installed or configured during setup, please do this manually."
      else
        echo "Error: Invalid option argument $PYTHON_INST" >&2
        exit_abnormal
      fi
      ;;
    h)  # display Help
      exit_abnormal
      ;;
    :)  # If expected argument omitted:
      echo "Error: Option -${OPTARG} requires an argument."
      exit_abnormal
      ;;
    *)  # If unknown (any other) option:
      echo "Error: Invalid option -$OPTARG" >&2
      exit_abnormal
      ;;
  esac
done

# Standard update
sudo apt update -y
sudo apt upgrade -y

# Install git
sudo apt install -y git
#sudo apt-get install -y curl openssh-server ca-certificates 

# Temp solution to avoid prompt while sourcecode is closed-source during alpha phase
#sudo cp -r /media/Shared/ssh/.ssh .
#sudo chown -R $USER ~/.ssh
#chmod -R 0700 ~/.ssh
#echo 'Host gitlab.com' >> ~/.ssh/config
#echo '    StrictHostKeyChecking no' >> ~/.ssh/config
#git config --global user.email "you@example.com"
#git config --global user.name "Your Name"

# Check if VirtualLab directory exists in $HOME
cd ~
if [ -d "$VL_DIR" ]; then
  # Take action if $VL_DIR exists. #
  echo
  echo "Skipping mkdir as ${VL_DIR} already exists"
  #source $VL_DIR/.VLprofile
else
  ###  Control will jump here if $VL_DIR does NOT exist ###
  echo
  echo "Creating ${VL_DIR} directory"
  sudo -u ${SUDO_USER:-$USER} mkdir ${VL_DIR}
fi

if grep -q $VL_DIR ~/.bashrc; then
  echo "VirtualLab is already in PATH"
else
  # Adding VirtualLab to PATH
  echo "Adding VirtualLab to PATH"
  sudo -u ${SUDO_USER:-$USER} echo 'export PATH="'$VL_DIR':$PATH"'  >> ~/.bashrc
#  sudo -u ${SUDO_USER:-$USER} echo 'export PATH="'$VL_DIR':$PATH"'  >> $VL_DIR/.VLprofile
  export PATH="'$VL_DIR':$PATH"
fi

### Download latest VirtualLab code
cd $VL_DIR
#sudo -u ${SUDO_USER:-$USER} git init
#sudo -u ${SUDO_USER:-$USER} git pull git@gitlab.com:ibsim/virtuallab.git
#git clone https://gitlab.com/ibsim/virtuallab.git
### Only download src with no history
#sudo -u ${SUDO_USER:-$USER} git pull --depth 1 git@gitlab.com:ibsim/virtuallab.git
### Must use git clone if planning to commit changes.
### Can comment out 'git init' above if using this.
if test -d ".git"; then
  sudo -u ${SUDO_USER:-$USER} git pull git@gitlab.com:ibsim/virtuallab.git
else
  sudo -u ${SUDO_USER:-$USER} git clone git@gitlab.com:ibsim/virtuallab.git .
fi

source VLconfig.py
# Change permissions on setup and run scripts
#chmod 755 Setup.sh
#chmod 755 Test_VL.py

# Run initial VirtualLab setup (including salome install)
if [ "$PYTHON_INST" == "y" ]; then
  echo "Installing python"
  source Scripts/Install/Install_python.sh
elif [ "$PYTHON_INST" == "c" ]; then
  echo "Installing/configuring conda"
  source Scripts/Install/Install_python.sh
else
  echo "Skipping python installation"
fi

cd $VL_DIR
if [ "$SALOME_INST" == "y" ]; then
  echo "Installing salome"
  source Scripts/Install/Install_Salome.sh
else
  echo "Skipping salome installation"
  echo
fi

# Currently can only run test as SU (therefore output files protected)
#sudo -u ubuntu python3 Test_VL.py
#sudo -u ${SUDO_USER:-$USER} ./Test_VL.py
#Test_VL.py
# Need to add test to check results
# Remove test files created by SU
#rm -r ~/VirtualLab/Training/
