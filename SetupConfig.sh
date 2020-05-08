#!/bin/bash

source VLconfig.sh

rm VLconfig.py
for i in ${!var[@]}; do
  echo ${var[$i]}'="'"${!var[i]}"'"' >> VLconfig.py
done

