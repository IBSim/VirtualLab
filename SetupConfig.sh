#!/bin/bash
if [ -f ~/.profile ]; then source ~/.profile; fi

cd $VL_DIR
CONFIG_FNAME=VLconfig_DEFAULT.sh
usage() {
  echo
  echo "Usage:"
  echo " $0 [ -f "'$FNAME'" ]"
  echo
  echo "A script to configure VirtualLab installation."
  echo
  echo "Options:"
  echo "   '-f "'$FNAME'"' Where "'$FNAME'" is the name of the python config file."
  echo
  echo "Default behaviour is to setup using VLconfig_DEFAULT.sh."
}
exit_abnormal() {
  usage
  exit 1
}
while getopts ":f:h" options; do 
  case "${options}" in
    f)
      CONFIG_FNAME=${OPTARG}
      if test ! -f "$CONFIG_FNAME" ; then
        echo 'The file "'$CONFIG_FNAME'" does not exist.'
        exit 1
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
  echo "Error: Too many arguments were entered, please check usage and flags."
  echo "Remaining arguments: \"$@\"."
  exit_abnormal
fi

### Run VLconfig bash script
source $CONFIG_FNAME

### Output list of config values to VLconfig python file
echo "Creating VLconfig.py in $VL_DIR."
echo "#!/usr/bin/env python3" > $VL_DIR/VLconfig.py
for i in ${!var[@]}; do
  echo ${var[$i]}'="'"${!var[i]}"'"' >> $VL_DIR/VLconfig.py
done
echo

