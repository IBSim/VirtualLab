#!/usr/bin/env python3

### This list of config variables is being ready by python and bash.
###  - Variables included here are default values. Others are untested.
###  - Keep definitions cross compatible where possible.
###  - Do not leave spaces next to equal signs.
###  - Avoid using bash or python variables within variables which will be 
### called by both environments. Workaround is to duplicate vars for each lang.

### PATH to VirtualLab directory.
### This is to enable running VirtualLab from locations other than the source 
### top directory.
### If left commented it is only possible to run from $VL_DIR
#VL_DIR = "/PATH/TO/VirtualLab/HERE"
VL_DIR_NAME="VirtualLab"
VL_DIR_py="os.path.expanduser('~')+'/'+VL_DIR_NAME"
VL_DIR_bsh="$HOME/$VL_DIR_NAME"

### Version of conda to download and install with
### wget https://repo.anaconda.com/archive/"$CONDAVER"
CONDAVER='Anaconda3-2020.02-Linux-x86_64.sh'

### Salome-Meca download, install and config variables
### Installation location
SALOMEDIR='/opt/SalomeMeca'
### Version number in download filename
SALOMEVER='salome_meca-2019.0.3-1-universal'
### Version number in unpacked directory
SALOMEBIN='V2019.0.3_universal'

### PATH to various directories required as in/out for VirtualLab.
### If left commented default behaviour is to locate in $VL_DIR
#InputDir = ""
#MaterialsDir = ""
#RunFilesDir = ""
OutputDir="$VLDir/Output"
