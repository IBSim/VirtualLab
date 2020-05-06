#!/bin/bash

VL_DIR_NAME="VirtualLab"
VL_DIR="$HOME/$VL_DIR_NAME"

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
  echo "Skipping mkdir as ${VL_DIR} already exists"
  source $VL_DIR/.VLprofile
else
  ###  Control will jump here if $VL_DIR does NOT exist ###
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

# Download latest VirtualLab code
cd $VL_DIR
#sudo -u ${SUDO_USER:-$USER} git init
#sudo -u ${SUDO_USER:-$USER} git pull git@gitlab.com:ibsim/virtuallab.git
#git clone https://gitlab.com/ibsim/virtuallab.git
# Only download src with no history
#sudo -u ${SUDO_USER:-$USER} git pull --depth 1 git@gitlab.com:ibsim/virtuallab.git
# Must use git clone if planning to commit changes.
# Can comment out 'git init' above if using this.
if test -d ".git"; then
  sudo -u ${SUDO_USER:-$USER} git pull git@gitlab.com:ibsim/virtuallab.git
else
  sudo -u ${SUDO_USER:-$USER} git clone git@gitlab.com:ibsim/virtuallab.git .
fi

# Change permissions on setup and run scripts
#chmod 755 Setup.sh
#chmod 755 Test_VL.py

# Run initial VirtualLab setup (including salome install)
source Scripts/Install/Install_python.sh
cd $VL_DIR
source Scripts/Install/Install_Salome.sh

# Currently can only run test as SU (therefore output files protected)
#sudo -u ubuntu python3 Test_VL.py
#sudo -u ${SUDO_USER:-$USER} ./Test_VL.py
#Test_VL.py
# Need to add test to check results
# Remove test files created by SU
#rm -r ~/VirtualLab/Training/
