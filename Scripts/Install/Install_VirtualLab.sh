#!/bin/bash
set -e
################################################################################
##########    Main /install Script for VirtualLab ##############################
################################################################################

################################################################################
#                        Define Default Variables     
################################################################################
USER_HOME=$(eval echo ~"${SUDO_USER}")
#sudo -s eval 'echo ${SUDO_USER}'
if [ -f "$USER_HOME"/.VLprofile ]; then source "$USER_HOME"/.VLprofile; fi

echo
echo "Starting installation of VirtualLab."
echo

#: <<'END'
### Default location to install VirtualLab if no flag.
VL_DIR="$USER_HOME/VirtualLab"
SKIP=n
PYTHON_INST="n"
SALOME_INST="n"
ERMES_INST="n"
CAD2VOX_INST="n"
GVXR_INST="n"

ALL="n"
ASTER_SUBDIR="/V2019.0.3_universal/tools/Code_aster_frontend-20190/bin/as_run"

################################################################################
#                    Useful Functions
################################################################################

usage() {
  echo
  echo "Usage:"
  echo " $0 [-d <path>] [-A] [-P {y/c/n}] [-S \"{y/n} <path>\"] [-E {y/n}] [-C {y/n}] [-G {y/n}]"
  echo
  echo "A script to install VirtualLab and its dependencies."
  echo
  echo "Options:"
  echo "   '-d <path>' Specify the installation path for VirtualLab"
  echo "   '-A ' Install all packages. Note: This option repects other" 
  echo "    options so you can explicitly opt-out of packages you don't want by" 
  echo "    setting the appropriate arguments. For example -A -E n will install"
  echo "    everything except ERMES (ensure you put option A first)."
  echo "    Also note: This option simply sets the argument to y for all cases"
  echo "    Thus if you need other options eg. using conda for -P or setting"
  echo "    custom path for Salome you will need to use their specific arguments."
  echo "    "
  echo "   '-P y' Install python using local installation"
  echo "   '-P c' Install python using conda environment"
  echo "   '-P n' Do not install python"
  echo "   '-S \"y <path>\"' Install Salome-Meca at <path> location"
  echo "   '-S y' Install Salome-Meca at default location /opt/SalomeMeca"
  echo "   '-S n' Do not install Salome-Meca"
  echo "   '-E y' Install ERMES at default location /opt/ERMES"
  echo "   '-E n' Do not install ERMES"
  echo "   '-C y' Install Cad2Vox"
  echo "   '-C n' Do not install Cad2Vox"
  echo "   '-G y' Install GVXR"
  echo "   '-G n' Do not install GVXR"
  echo "   '-y' Skip install confirmation dialogue."
  echo
  echo "Default behaviour is to not install python, salome or ERMES."
  echo "Default install locations are: VirtualLab in the user's home directory,"
  echo "salome in '/opt/SalomeMeca', ERMES in '/opt/ERMES', python/conda in the"
  echo "recommended locations."
}

exit_abnormal() {
  usage
  exit 1
}

check_for_conda() {
    # Check if Conda is installed and if so define a flag to use use conda and
    # activate the vitrual env. If conda is not found, prerequisites are
    # assumed installed in local python
    search_var="anaconda*"
    search_var2="miniconda*"
    conda_dir=$(eval find $USER_HOME -maxdepth 1 -type d -name "$search_var")
    mini_conda_dir=$(eval find $USER_HOME -maxdepth 1 -type d -name "$search_var2")
    if [[ -f $conda_dir/bin/conda ]]; then
	eval "$($conda_dir/bin/conda shell.bash hook)"
	USE_CONDA=True
    elif [[ -f $mini_conda_dir/bin/conda ]]; then
	eval "$($mini_conda_dir/bin/conda shell.bash hook)"
	USE_CONDA=True
    else
	USE_CONDA=False	
    fi

  # If conda found activate environment
  
  if hash conda 2>/dev/null; then
    CONDAENV="$(basename -- $VL_DIR)"
    conda activate $CONDAENV
  fi
  
  }
# Create a simple pretty banner for announcing build stages
# Usage: banner "title" "colour" "symbol"
# Title: single line sting to print as a message.
# Colour: can be White, red, green or blue. Anything else defaults to white.
# Symbol: any Single character to repeat for the border.
# eg. banner 'Hello' 'red' '*'
function banner() {
    case ${2} in
    white)
        colour=7
        ;;
    red)
        colour=1
        ;;
    green)
        colour=2
        ;;
    blue)
        colour=4
        ;;
    *)
        colour=7
        ;;
    esac
    local msg="${3} ${1} ${3}"
    local edge=$(echo "${msg}" | sed "s/./${3}/g")
    tput setaf ${colour}
    tput bold
    echo "${edge}"
    echo "${msg}"
    echo "${edge}"
    tput sgr 0
    echo
  }

function exists_in_list() {
    LIST=$1
    DELIMITER=$2
    VALUE=$3
    echo $LIST | tr "$DELIMITER" '\n' | grep -F -q -x "$VALUE"
}
################################################################################
#                    Parse CMD Arguments
################################################################################
if [[ $EUID -ne 0 ]]; then
   echo "This installation script must be run as root"
   echo 'Re-run with "sudo ./Install_VirtualLab.sh {options}".'
   exit_abnormal
fi
while getopts "d:AP:S:E:C:G:yh" options; do
  case "${options}" in
    d)
      VL_DIR=$(readlink -m ${OPTARG})
      echo " - VirtualLab will be installed in '$VL_DIR'."
      ;;
    A)
	ALL="y"
	PYTHON_INST="y"
	SALOME_INST="y"
	ERMES_INST="y"
	CAD2VOX_INST="y"
      ;;
    P)
      PYTHON_INST=${OPTARG}
      if [ "$PYTHON_INST" == "y" ]; then
### skip displaying message if using -A to avoid doubling up install messages.
      	if [ "$ALL" != "y" ]; then
        	echo " - Python will be installed/updated and configured as part of VirtualLab install."
	fi
      
      elif [ "$PYTHON_INST" == "c" ]; then
        echo " - Conda will be installed/updated and configured as part of VirtualLab install."
      elif [ "$PYTHON_INST" == "n" ]; then
        echo " - Python/conda will not be installed or configured during setup."
        echo "   please do this manually, or by sourcing Install_python.sh."
      else
        echo "Error: Invalid option argument $PYTHON_INST" >&2
        exit_abnormal
      fi
      ;;
    S)
      SALOME_INST=${OPTARG}
      if [[ "$SALOME_INST" == "y" ]]; then
	### skip displaying message if using -A to avoid doubling up install messages.
         if [ $ALL != "y" ]; then
            echo " - Salome-Meca will be installed in the default directory and configured as part of VirtualLab install."
     	 fi

      elif [[ "$SALOME_INST" == "n" ]]; then
        echo " - Salome-Meca will not be installed or configured during setup,"
        echo "   please do this manually or by sourcing Install_Salome.sh."
      elif [[ "$SALOME_INST" == "y"* ]]; then
        set -f # disable glob
	IFS=' ' # split on space characters
        array=($OPTARG) # use the split+glob operator
        if [[ ! ${#array[@]} == 2 ]]; then
          echo "The number of arguments entered for option -S is ${#array[@]}." >&2
          echo "The number expected is 2, i.e. [-S \"y <path>\"]" >&2
          echo "or [-S {y/n}] if no path is specified." >&2
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
    E)
      ERMES_INST=${OPTARG}
      if [ "$ERMES_INST" == "y" ]; then
      ### skip displaying message if using -A to avoid doubling up install messages.
         if [ $ALL != "y" ]; then
        	echo " - ERMES will be installed in the default directory and configured as part of VirtualLab install."
	 fi

      elif [ "$ERMES_INST" == "n" ]; then
        echo " - ERMES will not be installed or configured during setup,"
        echo "   please do this manually or by sourcing Install_ERMES.sh."
      else
        echo "Error: Invalid option argument $ERMES_INST" >&2
        exit_abnormal
      fi
      ;;
    C)
	CAD2VOX_INST=${OPTARG}
      if [ "$CAD2VOX_INST" == "y" ]; then
      ### skip displaying message if using -A to avoid doubling up install messages.
         if [ $ALL != "y" ]; then
        	echo " - Cad2Vox will be installed in the default directory and configured as part of VirtualLab install."
         fi

      elif [ "$CAD2VOX_INST" == "n" ]; then
        echo " - Cad2Vox will not be installed or configured during setup,"
        echo "   please do this manually or by sourcing Install_Cad2Vox.sh."
      else
        echo "Error: Invalid option argument $CAD2VOX_INST" >&2
        exit_abnormal
      fi
      ;;
    G)
	GVXR_INST=${OPTARG}
      if [ "$GVXR_INST" == "y" ]; then
      ### skip displaying message if using -A to avoid doubling up install messages.
         if [ $ALL != "y" ]; then
        	echo " - GVXR will be installed in the default directory and configured as part of VirtualLab install."
         fi

      elif [ "$GVXR_INST" == "n" ]; then
        echo " - GVXR will not be installed or configured during setup,"
        echo "   please do this manually or by sourcing Install_Cad2Vox.sh."
      else
        echo "Error: Invalid option argument $GVXR_INST" >&2
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
### All this just to make -A respect other options and not duplicate confirmation messages.
if [ "$ALL" == "y" ]; then
	if [ "$PYTHON_INST" == "y" ]; then
        	echo " - Python will be installed/updated and configured as part of VirtualLab install."
	fi

	if [[ "$SALOME_INST" == "y" ]]; then
		if ! [[ -v STRING_TMP ]]; then
			echo " - Salome-Meca will be installed in the default directory and configured as part of VirtualLab install."
		fi
	fi

	if [ "$ERMES_INST" == "y" ]; then
        	echo " - ERMES will be installed in the default directory and configured as part of VirtualLab install."
     	fi
	
	if [ "$CAD2VOX_INST" == "y" ]; then
        	echo " - Cad2Vox will be installed in the default directory and configured as part of VirtualLab install."
	fi
	
	if [ "$GVXR_INST" == "y" ]; then
        	echo " - GVXR will be installed in the default directory and configured as part of VirtualLab install."
	fi
fi
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
    banner "Make it so!" "green" '*'
  else
    echo "Exiting VirtualLab installation/configuration."
    exit
  fi
fi

################################################################################
#                   Actual Install
################################################################################

### Standard update
sudo apt update -y
sudo apt upgrade -y

### Install requirements
sudo apt install -y git

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
  # Add VL_DIR to VLProfile so that different parts of instal can be run seperately
  sudo -u ${SUDO_USER:-$USER} echo 'VL_DIR="'$VL_DIR'"' >> $USER_HOME/.VLprofile

  sudo -u ${SUDO_USER:-$USER} echo 'if [[ ! $PATH =~ "'$VL_DIR'" ]]; then' >> $USER_HOME/.VLprofile
#  sudo -u ${SUDO_USER:-$USER} echo '  export PATH="'$VL_DIR':$PATH"'  >> $USER_HOME/.VLprofile
  sudo -u ${SUDO_USER:-$USER} echo '  export PATH="'$VL_DIR'/bin:$PATH"'  >> $USER_HOME/.VLprofile
  sudo -u ${SUDO_USER:-$USER} echo 'fi'  >> $USER_HOME/.VLprofile

  export PATH="$VL_DIR/bin:$PATH"
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
if test -d ".git"; then
    sudo -u ${SUDO_USER:-$USER} git fetch
    sudo -u ${SUDO_USER:-$USER} git reset --hard HEAD
    sudo -u ${SUDO_USER:-$USER} git merge '@{u}'
else
  sudo -u ${SUDO_USER:-$USER} git clone https://gitlab.com/ibsim/virtuallab.git .
  sudo chown -R ${SUDO_USER:-$USER} $VL_DIR
fi
#END

sudo -u ${SUDO_USER:-$USER} git checkout BT-Container
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
  banner "Installing python" "green" "*"
  source $VL_DIR/Scripts/Install/Install_python.sh
elif [ "$PYTHON_INST" == "c" ]; then
  banner "Installing/configuring conda" "green" "*"
  source $VL_DIR/Scripts/Install/Install_python.sh
else
  banner "Skipping python installation" "blue" "*"
fi

check_for_conda

### Install salome if flagged
if [ "$SALOME_INST" == "y" ]; then
  banner "Installing salome" "green" "*"
  source $VL_DIR/Scripts/Install/Install_Salome.sh
else
  banner "Skipping salome installation" "blue" "*"
fi

echo
### Install ERMES if flagged
if [ "$ERMES_INST" == "y" ]; then
  banner "Installing ERMES" "green" "*"
  source $VL_DIR/Scripts/Install/Install_ERMES.sh
else
  banner "Skipping ERMES installation" "blue" "*"
fi

### Install Cad2Vox if flagged
if [ "$CAD2VOX_INST" == "y" ]; then
  banner "Installing Cad2Vox" "green" "*"
  source $VL_DIR/Scripts/Install/Install_Cad2Vox.sh
else
  banner "Skipping Cad2Vox installation" "blue" "*"
fi
### Install GVXR if flagged
if [ "$GVXR_INST" == "y" ]; then
  banner "Installing GVXR" "green" "*"
  source $VL_DIR/Scripts/Install/Install_GVXR_brew.sh
else
  banner "Skipping GVXR installation" "blue" "*"
fi

sudo chown -R ${SUDO_USER:-$USER} $VL_DIR

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
banner "Finished installing and configuring VirtualLab." "green" "*"
echo
echo "Usage:"
echo " VirtualLab [ -f "'$FPATH'" ]"
echo
echo "Options:"
echo "   '-f "'$FPATH'"' Where "'$FPATH'" is the path of the python run file."
echo "   '-h Display this help menu."
echo
echo "Default behaviour is to exit if no "'$FPATH'" is given"
echo
set +e
#END
