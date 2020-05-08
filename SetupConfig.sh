#!/bin/bash

source VLconfig.sh
FILE="VLconfig.py"
if test -f "$FILE"; then
    rm $FILE
fi
for i in ${!var[@]}; do
  echo ${var[$i]}'="'"${!var[i]}"'"' >> VLconfig.py
done

