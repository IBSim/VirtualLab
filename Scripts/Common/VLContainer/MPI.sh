#!/bin/bash
command=$1
VL_hostname=$2
VL_port=$3
VL_DIR=$4

# slightly complex method of getting the host name, but environment variables 
# are not updated when MPI is launched, so we can't use $HOST/$HOSTNAME
temp_dir=$(mktemp -d)
host_file=$temp_dir/host
python3 $VL_DIR/VL_server2.py host $host_file
host=`cat $host_file`
echo $host

if [ $host = $VL_hostname ] ; then
    # task is being run on the same node as what the VirtualLab run file is run on
    $command $VL_port
else
    # task is run on a diffrent node so must set up a new tcp port to run on
    
    port_file=$temp_dir/port
    # make server and write port number to temp_file
    python3 $VL_DIR/VL_server2.py run $port_file & 
    while true ; do
        if [ -f $temp_file ] ; then 
            port=`cat $temp_file`
            break 
        fi  
        done
    $command $port
fi

rm -r $temp_dir