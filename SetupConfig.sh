#!/bin/bash
source VLconfig.sh

echo "#!/usr/bin/env python3" > VLconfig.py
for i in ${!var[@]}; do
  echo ${var[$i]}'="'"${!var[i]}"'"' >> VLconfig.py
done

