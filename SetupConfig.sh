#!/bin/bash
configfile_sh=$1
configfile_py=$2

source $configfile_sh

#FILE="VLconfig.py"
#if test -f "$FILE"; then
#    rm $FILE
#fi

for i in ${!var[@]}; do
  echo ${var[$i]}'="'"${!var[i]}"'"' >> $configfile_py
done

