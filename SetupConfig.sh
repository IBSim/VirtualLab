#!/bin/bash
if [ -f ~/.profile ]; then source ~/.profile; fi

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

echo $CONFIG_FNAME
source $CONFIG_FNAME

echo "#!/usr/bin/env python3" > VLconfig.py
for i in ${!var[@]}; do
  echo ${var[$i]}'="'"${!var[i]}"'"' >> VLconfig.py
done

