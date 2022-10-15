#!/bin/bash
set -e
USER_HOME=$(eval echo ~${SUDO_USER})
if [ -f $USER_HOME/.VLprofile ]; then source $USER_HOME/.VLprofile; fi


#########################
### This script is used to install/configure python/conda and its dependencies.
### It first attempts to detect whether it is already installed.
### For VirtualLab, the default config values are as below.
### These can be changed in $VL_DIR/VLconfig_DEFAULT.sh if needed.
### CONDA_VER='Anaconda3-2020.02-Linux-x86_64.sh'
### CONDAENV=$VL_DIR_NAME
#########################

### If configuring conda, use the name of the directory where VirtualLab is
### installed as the name of the conda environment. By default this is 'VirtualLab'.
CONDAENV=$(basename "$VL_DIR")
### By default don't install conda unless triggered by flag
CONDA_INST="n"

### Get flags to install python locally or in conda env.
usage() {
  echo "Usage:"
  echo " $0 [ -P {y/c/n} ]"
  echo
  echo "A script to install/setup python for VirtualLab"
  echo
  echo "Options:"
  echo "   '-P y' Install python using local installation"
  echo "   '-P c' Install python using conda envrionment"
  echo "   '-P n' Do not install python"
  echo
  echo "Default behaviour is to not install python."
}
exit_abnormal() {
  usage
  exit 1
}
while getopts ":P:" options; do
  case "${options}" in
    P) ### If P option triggered
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

### Standard update
sudo apt update -y
sudo apt upgrade -y
sudo apt install -y build-essential

source "$VL_DIR/VLconfig.py" # Enables this script to be run seperately

if [ "$PYTHON_INST" == "y" ]; then
  ### Install python and required packages
  sudo apt install -y python3.8
  sudo apt install -y python3-pip
  #sudo apt install -y python3-sphinx
  sudo pip3 install --user -U sphinx
  sudo -u ${SUDO_USER:-$USER} pip3 install --user furo sphinx-rtd-theme==0.4.3 sphinxcontrib-bibtex==1.0.0
  sudo -u ${SUDO_USER:-$USER} pip3 install --user -r $VL_DIR/requirements.txt
  sudo -u ${SUDO_USER:-$USER} pip3 install --user scikit-learn==0.24.1
  sudo -u ${SUDO_USER:-$USER} pip3 install --user -U --no-deps iapws==1.4

  # install pyina (uses MPI)
  sudo apt install -y mpich
  sudo -u ${SUDO_USER:-$USER} pip3 install --user mpi4py==3.0.3 dill==0.3.3 pox==0.2.9
  sudo -u ${SUDO_USER:-$USER} pip3 install --user -U --no-deps pyina==0.2.4
  # sourcing profile adds $HOME/.local/bin to $PATH for this terminal.
  # TODO: This is a temporary solution for this terminal, it is only permanent by
  # logging out and back in. Or could add this in to VLprofile?
  source ~/.profile

#  sudo -u ${SUDO_USER:-$USER} pip3 install --user numpy scipy matplotlib h5py sphinx-rtd-theme sphinxcontrib-bibtex
#  sudo -u ${SUDO_USER:-$USER} pip3 install --user iapws pathos==0.2.7

  ### Add $VL_DIR to $PYTHONPATH in python env and current shell
  if grep -q PYTHONPATH='$PYTHONPATH'$VL_DIR $USER_HOME/.VLprofile; then
    echo "Reference to VirtualLab PYTHONPATH found in ~/.VLprofile"
    echo "Therefore, not adding again."
  else
    echo "Adding $VL_DIR to PYTHONPATH"
    sudo -u ${SUDO_USER:-$USER} echo 'export PYTHONPATH=$PYTHONPATH'$VL_DIR''  >> $USER_HOME/.VLprofile
    export PYTHONPATH=$PYTHONPATH$VL_DIR

    ### ~/.bashrc doesn't get read by subshells in ubuntu.
    ### Workaround: store additions to env PATH in ~/.VLprofile & source in bashrc.
    STRING_TMP="if [ -f ~/.VLprofile ]; then source ~/.VLprofile; fi"
    if [[ ! $(grep -F "$STRING_TMP" $USER_HOME/.bashrc | grep -F -v "#$STRING") ]]; then
      echo $STRING_TMP >> $USER_HOME/.bashrc
    fi
  fi
elif [ "$PYTHON_INST" == "c" ]; then
  ### Install conda dependencies
  sudo apt install -y libgl1-mesa-glx libegl1-mesa libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6

  ### Check if Conda is installed
  search_var=anaconda*
  conda_dir=$(eval find $USER_HOME -maxdepth 1 -type d -name "$search_var")
  if [[ -f $conda_dir/bin/conda ]]; then
    eval "$($conda_dir/bin/conda shell.bash hook)"
  else
    search_var=miniconda*
    conda_dir=$(eval find $USER_HOME -maxdepth 1 -type d -name "$search_var")
    if [[ -f $conda_dir/bin/conda ]]; then
      eval "$($conda_dir/bin/conda shell.bash hook)"
    fi
  fi

  ### Test to check if conda already exists in current shell's PATH
  if hash conda 2>/dev/null; then
    ### If exists, do nothing
    echo
    echo "Conda is already installed."
    echo "Skipping conda installation."
  else
    ### Otherwise download and install conda
    conda_dir=$USER_HOME/anaconda3
    echo
    cd $USER_HOME
    if test ! -f "$CONDA_VER"; then
      echo "Proceeding to download conda in $USER_HOME"
      echo "Downloading https://repo.anaconda.com/archive/"$CONDA_VER""
      sudo -u ${SUDO_USER:-$USER} wget https://repo.anaconda.com/archive/"$CONDA_VER"
    fi
    echo "Proceeding to install conda in ${conda_dir}"
    sudo -u ${SUDO_USER:-$USER} bash $CONDA_VER -b -p ${conda_dir}
    eval "$(${conda_dir}/bin/conda shell.bash hook)"
    conda init
    sudo -s -u ${SUDO_USER} source $VL_DIR/Scripts/Install/conda_init.sh
    export PATH=${conda_dir}/bin:$PATH
    source $USER_HOME/.VLprofile
    ### Test conda
    if hash conda 2>/dev/null; then
      echo "Conda succesfully installed"
      echo
      ### rm download if installed
    else
      echo "There has been a problem installing Conda"
      echo "Check error messages, try to rectify then rerun this script"
      return # exit closes the terminal as it's sourced
    fi
    #conda --version
  fi
  conda update -n base -c defaults conda -y
  if test ! -d "${conda_dir}/envs/$CONDAENV"; then
    echo "Creating Conda env $CONDAENV"
    conda create -n $CONDAENV python=3.8 -y
  fi

  OS_v=$(eval lsb_release -r -s)
  if [[ $OS_v == "20.04" ]]; then
    if test ! -d "${conda_dir}/envs/python2"; then
      echo "OS is Ubuntu $OS_v, which doesn't have python2 installed as default."
      echo "A python2 environment is also being created to install Salome_Meca."
      echo "Creating Conda end python2."
      conda create -n python2 python=2.7 -y
    fi
  fi

  ### Install conda packages
  conda activate $CONDAENV
  conda config --append channels conda-forge

  conda install -y sphinx
  conda install -y furo sphinx_rtd_theme=0.4.3 sphinxcontrib-bibtex=1.0.0
  conda install -y --file $VL_DIR/requirements.txt
  conda install -y scikit-learn=0.24.1 iapws=1.4

  sudo chown $SUDO_USER:$SUDO_USER -R ${conda_dir}/envs/$CONDAENV
  sudo chmod -R 0755 ${conda_dir}/envs/$CONDAENV
  sudo chown -R 1000:1000 ${conda_dir}/pkgs/cache
  if test -e $USER_HOME/.cache/pip; then
    sudo chown -R 1000:1000 $USER_HOME/.cache/pip
  fi
  ### Install python and required packages
  sudo apt install -y python3-pip

  # install pyina (uses MPI)
  conda install pip
  sudo apt install -y mpich
  sudo -u ${SUDO_USER:-$USER} env "PATH=$PATH" pip install mpi4py dill pox
  sudo -u ${SUDO_USER:-$USER} env "PATH=$PATH" pip install -U pyina
  source ~/.profile # This adds $HOME/.local/bin to $PATH which is needed by pyina

  echo "Finished creating Conda env $CONDAENV"

  ### Add $VL_DIR to $PYTHONPATH in Conda env and current shell
  PYV=`python -V`
  PYV2=${PYV#* }
  PYV=${PYV2%.*}
  PATH_FILE=${conda_dir}/envs/$CONDAENV/lib/python$PYV/site-packages/$CONDAENV.pth
  if test -f "$PATH_FILE"; then
    echo "VirtualLab PYTHONPATH found in Conda env."
    echo
  else
    echo "Adding $VL_DIR to PYTHONPATH in Conda env."
    echo
    sudo -u ${SUDO_USER:-$USER} echo $VL_DIR >> $PATH_FILE
    export PYTHONPATH=$PYTHONPATH$VL_DIR
  fi
  echo "If conda was not previously installed you will need to open a new"
  echo "terminal to activate it or run the following command in this terminal:"
  echo 'eval "$(${conda_dir}/bin/conda shell.bash hook)"'
  echo
else
  echo "Skipping python installation"
  exit 1
fi

echo
### Build VirtualLab documentation using sphinx
source $VL_DIR/Scripts/Install/Install_docs.sh


#: <<'END'
#END
