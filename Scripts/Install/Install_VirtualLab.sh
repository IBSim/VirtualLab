#!/bin/bash
USER_HOME=$(eval echo ~${SUDO_USER})
#sudo -s eval 'echo ${SUDO_USER}'
if [ -f $USER_HOME/.VLprofile ]; then source $USER_HOME/.VLprofile; fi

echo
echo "Starting installation of VirtualLab."
echo

#: <<'END'
### Default location to install VirtualLab if no flag.
VL_DIR="$USER_HOME/VirtualLab"
SKIP=n
PYTHON_INST="n"
SALOME_INST="n"
ASTER_SUBDIR="/V2019.0.3_universal/tools/Code_aster_frontend-20190/bin/as_run"

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
  echo "   '-y' Skip install confirmation dialogue."
  echo
  echo "Default behaviour is to not install python or salome."
  echo "Default install locations are: VirtualLab in the user's home directory,"
  echo "salome in '/opt/SalomeMeca', python/conda in the recommended locations."
}
exit_abnormal() {
  usage
  exit 1
}
if [[ $EUID -ne 0 ]]; then
   echo "This installation script must be run as root"
   echo 'Re-run with "sudo ./Install_VirtualLab.sh {options}".'
   exit_abnormal
fi
while getopts ":d:P:S:yh" options; do 
  case "${options}" in
    d)
      VL_DIR=$(readlink -m ${OPTARG})
      echo " - VirtualLab will be installed in '$VL_DIR'."
      ;;
    P)
      PYTHON_INST=${OPTARG}
      if [ "$PYTHON_INST" == "y" ]; then
        echo " - Python will be installed/updated and configured as part of VirtualLab install."
      elif [ "$PYTHON_INST" == "c" ]; then
        echo " - Conda will be installed/updated and configured as part of VirtualLab install."
      else
        echo "Error: Invalid option argument $PYTHON_INST" >&2
        exit_abnormal
      fi
      ;;
    S)
      SALOME_INST=${OPTARG}
      if [[ "$SALOME_INST" == "y" ]]; then
        echo " - Salome-Meca will be installed in the default directory and configured as part of VirtualLab install."
      elif [[ "$SALOME_INST" == "y"* ]]; then
        set -f # disable glob
	IFS=' ' # split on space characters
        array=($OPTARG) # use the split+glob operator
        if [[ ! ${#array[@]} == 2 ]]; then
          echo "The number of arguments entered for option -S is ${#array[@]}."
          echo "The number expected is 2, i.e. [-S \"y <path>\"]"
          echo "or [-S {y/n}] if no path is specified."
          exit_abnormal
        fi
        SALOME_INST=${array[0]}
        STRING_TMP="${array[1]}"
        STRING_TMP=${STRING_TMP/'~'/$HOME}
        SALOME_DIR=$(readlink -m $STRING_TMP)
        echo " - Salome-Meca will be installed in '$SALOME_DIR' and configured as part of VirtualLab install."
      else
        echo "Error: Invalid option argument $SALOME_INST" >&2
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
if [ "$PYTHON_INST" == "n" ]; then
  echo " - Python/conda will not be installed or configured during setup,"
  echo "please do this manually."
fi
if [ "$SALOME_INST" == "n" ]; then
  echo " - Salome-Meca will not be installed or configured during setup,"
  echo "please do this manually."
fi
echo
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

#: <<'END'
### Standard update
sudo apt update -y
sudo apt upgrade -y

### Install requirements
sudo apt install -y git

### Temp solution to avoid prompt while sourcecode is closed-source during alpha phase
#sudo cp -r /media/Shared/ssh/.ssh .
#sudo chown -R $USER ~/.ssh
#chmod -R 0700 ~/.ssh
#echo 'Host gitlab.com' >> ~/.ssh/config
#echo '    StrictHostKeyChecking no' >> ~/.ssh/config
#git config --global user.email "you@example.com"
#git config --global user.name "Your Name"

### Check if VirtualLab directory exists in $HOME
cd $USER_HOME
if [ -d "$VL_DIR" ]; then
  #### If $VL_DIR exists don't do anything.
  echo
  echo "Skipping mkdir as ${VL_DIR} already exists."
else
  ### If not, create $VL_DIR
  echo
  echo "Creating ${VL_DIR} directory."
  sudo -u ${SUDO_USER:-$USER} mkdir ${VL_DIR}
fi

if [ -f "$USER_HOME/.VLprofile" ]; then
  sudo chown ${SUDO_USER} $USER_HOME/.VLprofile
  sudo chgrp ${SUDO_USER} $USER_HOME/.VLprofile
fi
### Check if VirtualLab is in PATH
if [[ $PATH =~ $VL_DIR ]]; then
  echo "VirtualLab is already in PATH."
else
  ### If not, add VirtualLab to PATH
  echo "Adding VirtualLab to PATH."
  
  sudo -u ${SUDO_USER:-$USER} echo 'if [[ ! $PATH =~ "'$VL_DIR'" ]]; then' >> $USER_HOME/.VLprofile
  sudo -u ${SUDO_USER:-$USER} echo '  export PATH="'$VL_DIR':$PATH"'  >> $USER_HOME/.VLprofile
  sudo -u ${SUDO_USER:-$USER} echo 'fi'  >> $USER_HOME/.VLprofile
  
  export PATH="$VL_DIR:$PATH"
fi

### ~/.bashrc doesn't get read by subshells in ubuntu.
### Workaround: store additions to env PATH in ~/.VLprofile & source in bashrc.
STRING_TMP="if [ -f ~/.VLprofile ]; then source ~/.VLprofile; fi"
if [[ ! $(grep -F "$STRING_TMP" $USER_HOME/.bashrc | grep -F -v "#$STRING") ]]; then 
  echo '' >> $USER_HOME/.bashrc
  echo '# Read in environment for VirtualLab' >> $USER_HOME/.bashrc
  echo $STRING_TMP >> $USER_HOME/.bashrc
fi

### Download latest VirtualLab code
cd $VL_DIR
### Only download src with no history
#sudo -u ${SUDO_USER:-$USER} git pull --depth 1 git@gitlab.com:ibsim/virtuallab.git
### Must use git clone if planning to commit changes.
if test -d ".git"; then
  sudo -u ${SUDO_USER:-$USER} git pull git@gitlab.com:ibsim/virtuallab.git
else
  sudo -u ${SUDO_USER:-$USER} git clone git@gitlab.com:ibsim/virtuallab.git .
fi
#END
### Run initial VirtualLab setup
echo

source "$VL_DIR/SetupConfig.sh"
#./SetupConfig.sh
#sudo -u ${SUDO_USER:-$USER} ./SetupConfig.sh

#sudo chown $(basename "${SUDO_USER}") $VL_DIR/VLconfig.py
sudo chown ${SUDO_USER} $VL_DIR/VLconfig.py
chmod 0755 $VL_DIR/VLconfig.py
source "$VL_DIR/VLconfig.py"

#: <<'END'
### Install/configure python/conda if flagged
if [ "$PYTHON_INST" == "y" ]; then
  echo "Installing python"
  source $VL_DIR/Scripts/Install/Install_python.sh
elif [ "$PYTHON_INST" == "c" ]; then
  echo "Installing/configuring conda"
  source $VL_DIR/Scripts/Install/Install_python.sh
  eval "$($HOME/anaconda3/bin/conda shell.bash hook)"
  conda activate $CONDAENV
else
  echo "Skipping python installation"
fi

echo
### Install salome if flagged
if [ "$SALOME_INST" == "y" ]; then
  echo "Installing salome"
  source $VL_DIR/Scripts/Install/Install_Salome.sh
else
  echo "Skipping salome installation"
fi

echo
### Build VirtualLab documentation using sphinx
source $VL_DIR/Scripts/Install/Install_docs.sh

: <<'END'
#commented out script
echo
### Build VirtualLab documentation using sphinx
echo "Building documentation"
cd $VL_DIR/docs
make clean
make html
sudo chown ${SUDO_USER} -R $VL_DIR/docs/build
sudo chgrp ${SUDO_USER} -R $VL_DIR/docs/build
sudo chmod -R 0755 $USER_HOME/anaconda3/envs/$CONDAENV
cd $VL_DIR
rm docs.html
sudo -u ${SUDO_USER:-$USER} ln -s docs/build/html/index.html docs.html
END

### Currently can only run test as SU (therefore output files protected)
#sudo -u ubuntu python3 Test_VL.py
#sudo -u ${SUDO_USER:-$USER} ./Test_VL.py
#Test_VL.py
### Need to add test to check results
### Remove test files created by SU
#rm -r ~/VirtualLab/Training/
#END
echo
echo "Finished installing and configutng VirtualLab."
echo
#END
