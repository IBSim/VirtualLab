#!/bin/bash
command=$1
VL_hostname=$2
VL_port=$3

if [ $HOSTNAME = $VL_hostname ] ; then
    # task is being run on the same node as what the VirtualLab run file is run on
    $command $VL_port
else
    # task is run on a diffrent node so must set up a new tcp port to run on
    temp_dir=$(mktemp -d)
    temp_file=$temp_dir/port
    # make server and write port number to temp_file
    python3 {vlab_dir}/VL_server2.py $temp_file & 
    while true ; do
        if [ -f $temp_file ] ; then 
            port=`cat $temp_file`
            rm -r $temp_dir
            break 
        fi  
        done
    $command $port
fi