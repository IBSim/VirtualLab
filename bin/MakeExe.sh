#!/bin/sh
# File which creates VirtualLab executable from VL_server.py
bindir="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
VLdir=$(dirname $bindir)

# create temp dir as pyinstaller has a lot of output 
tmpdir=$(mktemp -d)
cd $tmpdir

python3 $bindir'/MakeExe.py' -F $VLdir'/VL_server.py' -n VirtualLab # make executable

cp 'dist/VirtualLab' $bindir'/VirtualLab' # only copy what we need
rm -r $tmpdir # remove temp directory created

