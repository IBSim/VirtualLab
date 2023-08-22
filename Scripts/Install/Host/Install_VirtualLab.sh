#!/bin/bash

# Script for downloading the VirtualLab repo and configuriging its use on the host system. 

USER_HOME=$(eval echo ~"${USER}")

usage() {
  echo
  echo "Usage:"
  echo " $0 [-B {name}] [-I {python/conda/binary}] "
  echo
  echo "A script to install VirtualLab with default settings."
  echo
  echo "   '-B {Branch_name}' Install VirtualLab from Branch_name (default is master)"
  echo "   '-I {python/conda/binary/N}' How VirtualLab is installed, either using standard python (default), conda or pre-built binaries."
  echo "                                N will not install any new packages, giving the user more flexibility."
  echo "   '-d {directory}' Path to custom directory in which to install VirtuaLab"
  echo "   '-y' Skip install confirmation dialogue."
}

exit_abnormal() {
  usage
  exit 1
}

### Default values for parsed arguments.
BRANCH=master
VL_BINARY='python'
SKIP=""
INST_DIR=""
MESSAGE=true

################################################################################
#                    Parse CMD Arguments
################################################################################

while getopts "B:I:d:yhZ" options; do
  case "${options}" in
    B)
      BRANCH="${OPTARG}"
      ;;
    I) # how installation and binaries are created
      VL_BINARY="${OPTARG}"
      if ! ([ "$VL_BINARY" = "python" ] || [ "$VL_BINARY" = "conda" ] || [ "$VL_BINARY" = "binary" ] || [ "$VL_BINARY" = "N" ]) ; then
        echo "Error: Invalid argument $VL_BINARY for option I" >&2
        exit_abnormal      
      fi
      ;;
    d) 
      INST_DIR="-d ${OPTARG}"
      ;;
    y)  ### Skip install confirmation dialogue.
      SKIP=-y
      ;;
    h)  ### display Help
      exit_abnormal
      ;;
    Z) # flag to say if its been called by main install script so not to repeat messages
       # not for general use
      MESSAGE=false
      ;;
    :)  ### If expected argument omitted:
      echo "Error: Option -${OPTARG} requires an argument."
      exit_abnormal
      ;;
    *)
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

echo
echo "Installing VirtualLab"
echo "~~~~~~~~~~~~~~~~~~~~~"
echo

# provide information about install (unless its called by main)
if [ "$MESSAGE" = true ] ; then
  echo
  echo "VirtualLab will be installed from branch $BRANCH using $VL_BINARY."
  echo

  if ! ([ "$BRANCH" = "master" ] || [ "$BRANCH" = "dev" ]) ; then
    echo "############## Warning #############"
    echo
    echo "Errors may occur when installing from branches other than master or dev"
    echo
    echo "####################################"
  fi

  ### Double check with user that they're happy to continue.
  ### This is skippable with -y flag.
  if [[ ! "$SKIP" =~ "y" ]]; then
    echo
    read -r -p "Are you sure? [y/n] " response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
      echo "Make it so!"
    else
      echo "Exiting VirtualLab installation/configuration."
      exit
    fi
  fi
fi

if [ "$VL_BINARY" = "python" ] ; then
  # check python version?
  echo
  echo "Installing required packages"
  echo "#############################"
  echo

  pip3 install gitpython pyinstaller

elif [ "$VL_BINARY" = "conda" ] ; then
  echo
  echo "Creating conda environment named 'VirtualLab'"
  echo "This will need to be active to run VirtualLab."
  echo

  conda create -y -n VirtualLab python=3.9
  conda activate VirtualLab
  conda install -y -c conda-forge gitpython
  python3 -m pip install pyinstaller

elif [ "$VL_BINARY" = "N" ] ; then
  echo
  echo "No python packages installed. It is assumed that gitpython and pyinstaller are already available."
  echo "VirtualLab will be downloaded using python"
  echo

fi


cd $USER_HOME

if [ "$VL_BINARY" == "binary" ]; then
  ### Install VirtualLab using installation binary

  if [ "$BRANCH" == "master" ]; then
    BRANCH_bin=main
  else 
    BRANCH_bin=dev
  fi

  echo
  echo "Downloading installation binary from"
  echo "gitlab.com/ibsim/virtuallab_bin (branch $BRANCH_bin) "
  echo "#############################"
  echo

  # installation binary
  fname=Install_VirtualLab
  url="https://gitlab.com/ibsim/virtuallab_bin/-/raw/"$BRANCH_bin"/"$fname
  wget $url

  echo
  echo "Starting installation"
  echo "#####################"
  echo
  chmod 755 $fname
  ./$fname -B $BRANCH $INST_DIR $SKIP
  rm $fname

  echo
  echo "Downloading VirtualLab binary"
  echo "#############################"
  echo
  source .VLprofile
  cd $VL_DIR"/bin"
  fname=VirtualLab
  url="https://gitlab.com/ibsim/virtuallab_bin/-/raw/"$BRANCH_bin"/"$fname
  wget $url
  chmod +x VirtualLab
  cd $USER_HOME

else
  # install using python (including conda)

  echo
  echo "Starting installation"
  echo "#####################"
  echo
  fname="Vlab_install.py"
  url="https://gitlab.com/ibsim/virtuallab/-/raw/"$BRANCH"/Scripts/Install/Host/"$fname
  wget $url

  # Install VirtualLab
  python3 $fname -B $BRANCH $INST_DIR $SKIP
  rm $fname

  # # build VirtualLab binary
  # echo
  # echo "Building VirtualLab binary"
  # echo "##########################"  
  # echo
  # source .VLprofile # ensures VirtualLab bin directory is in $PATH
  #$VL_DIR"/bin/BuildVL"

fi

source .VLprofile # ensures VirtualLab is in $PATH

### Test to check if installation worked
if hash VirtualLab 2>/dev/null; then
  ### If exists
  echo
  echo "##########################################"
  echo
  echo "VirtualLab has been successfully installed"
  echo
  echo "##########################################"
  echo
  VirtuaLab
else
  ### VirtualLab still not installed
  echo
  echo "There has been a problem installing VirtualLab"
  echo "Check error messages, try to rectify then rerun this script"
  echo
  exit
  
fi
