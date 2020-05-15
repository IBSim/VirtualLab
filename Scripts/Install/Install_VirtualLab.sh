#!/bin/bash
if [ -f ~/.profile ]; then source ~/.profile; fi
echo

VL_DIR_NAME="VirtualLab"
VL_DIR="$HOME/$VL_DIR_NAME"

usage() {
  echo
  echo "Usage:"
  echo " $0 [-d <path>] [-P {y/c/n}] [-S \"{y/n} <path>\"]"
  echo
  echo "A script to install VirtualLab and its dependencies."
  echo
  echo "Options:"
  echo "   '-d <path>' Specify the installation path for VirtualLab"
  echo "   '-P y' Install python using local installation"
  echo "   '-P c' Install python using conda environment"
  echo "   '-P n' Do not install python"
  echo "   '-S \"y <path>\"' Install Salome-Meca at <path> location"
  echo "   '-S y' Install Salome-Meca at defauly location /opt/SalomeMeca"
  echo "   '-S n' Do not install Salome-Meca"
  echo
  echo "Default behaviour is to not install python or salome."
  echo "Default install locations are: VirtualLab in the user's home directory,"
  echo "salome in '/opt/SalomeMeca', python/conda in the recommended locations."
}
exit_abnormal() {
  usage
  exit 1
}
while getopts ":d:P:S:h" options; do 
  case "${options}" in
    d)
      VL_DIR=$(readlink -m ${OPTARG})
      VL_DIR_NAME=$(basename "$VL_DIR")
      echo "VirtualLab will be installed in '$VL_DIR'."
      ;;
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
      if [[ "$SALOME_INST" == "y" ]]; then
        echo "Salome-Meca will be installed in the default directory and configured as part of VirtualLab install."
      elif [[ "$SALOME_INST" == *"y"* ]]; then
        set -f # disable glob
	IFS=' ' # split on space characters
        array=($OPTARG) # use the split+glob operator
        if [[ ${#array[@]} > 2 ]]; then
          echo "The number of arguments entered for option -S is ${#array[@]}."
          echo "The max. number expected is 2, i.e. [-S \"{y/n} <path>\"]"
          echo "or [-S {y/n}] if no path is specified."
          exit_abnormal
        fi
        SALOME_INST=${array[0]}
        #echo "$SALOME_INST"
        STRING_TMP="${array[1]}"
        #echo "1. "$STRING_TMP
        STRING_TMP=${STRING_TMP/'~'/$HOME}
        #echo "2. "$STRING_TMP
        #echo "3. "$(readlink -m $STRING_TMP)
        SALOME_DIR=$(readlink -m $STRING_TMP)
        #echo "4. "$SALOME_DIR
        #echo "${array[1]}"
        echo "Salome-Meca will be installed in '$SALOME_DIR' and configured as part of VirtualLab install."
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

shift $(($OPTIND - 1))
if [[ $@ ]]; then
  echo
  echo "Error: Too many arguments were entered, please check."
  echo "Remaining arguments: \"$@\"."
  exit_abnormal
fi

#: <<'END'
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
  echo "Skipping mkdir as ${VL_DIR} already exists."
  #source $VL_DIR/.VLprofile
else
  ###  Control will jump here if $VL_DIR does NOT exist ###
  echo
  echo "Creating ${VL_DIR} directory."
  sudo -u ${SUDO_USER:-$USER} mkdir ${VL_DIR}
fi

if [[ $PATH =~ $VL_DIR ]]; then
  echo "VirtualLab is already in PATH."
else
  # Adding VirtualLab to PATH
  echo "Adding VirtualLab to PATH."
  sudo -u ${SUDO_USER:-$USER} echo 'export PATH="'$VL_DIR':$PATH"'  >> ~/.profile
#  sudo -u ${SUDO_USER:-$USER} echo 'export PATH="'$VL_DIR':$PATH"'  >> $VL_DIR/.VLprofile
  export PATH="'$VL_DIR':$PATH"
fi

# ~/.bashrc doesn't get read by subshells in ubuntu.
# Workaround: store additions to env PATH in ~/.profile & source in bashrc.
STRING_TMP="if [ -f ~/.profile ]; then source ~/.profile; fi"
if [[ ! $(grep -F "$STRING_TMP" ~/.bashrc | grep -F -v "#$STRING") ]]; then 
  echo $STRING_TMP >> ~/.bashrc
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

echo
sudo -u ${SUDO_USER:-$USER} ./SetupConfig.sh
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
#END
echo
echo "Finished installing and configuting VirtualLab."
echo


