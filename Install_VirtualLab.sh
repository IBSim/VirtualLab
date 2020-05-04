#!/bin/bash

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

# Check if VirtualLab directory exists in /home/$USER
cd ~
current_dir=$(pwd)
DIR="$current_dir/VirtualLab"
if [ -d "$DIR" ]; then
  # Take action if $DIR exists. #
  echo "Skipping mkdir as ${DIR} already exists"
  source ~/VirtualLab/.VLprofile
else
  ###  Control will jump here if $DIR does NOT exist ###
  echo "Creating ${DIR} directory"
  sudo -u ${SUDO_USER:-$USER} mkdir ${DIR}
fi

# Download latest VirtualLab code
cd ~/VirtualLab
sudo -u ${SUDO_USER:-$USER} git init
#sudo -u ${SUDO_USER:-$USER} git pull git@gitlab.com:ibsim/virtuallab.git
#git clone https://gitlab.com/ibsim/virtuallab.git
# Only download src with no history
sudo -u ${SUDO_USER:-$USER} git pull --depth 1 git@gitlab.com:ibsim/virtuallab.git
# Must use git clone if planning to commit changes.
# Can comment out 'git init' above if using this.
#sudo -u ${SUDO_USER:-$USER} git clone git@gitlab.com:ibsim/virtuallab.git .

# Change permissions on setup and run scripts
chmod 755 Setup.sh
chmod 755 Test_VL.py

# Run initial VirtualLab setup (including salome install)
source Setup.sh
cd ~/VirtualLab

# Currently can only run test as SU (therefore output files protected)
#sudo -u ubuntu python3 Test_VL.py
#sudo -u ${SUDO_USER:-$USER} ./Test_VL.py
./Test_VL.py
# Need to add test to check results
# Remove test files created by SU
#rm -r ~/VirtualLab/Training/
