#!/bin/bash
configfile=$1

source VLconfig.sh

#FILE="VLconfig.py"
#if test -f "$FILE"; then
#    rm $FILE
#fi

for i in ${!var[@]}; do
  echo ${var[$i]}'="'"${!var[i]}"'"' >> $configfile
done

