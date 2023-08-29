#!/bin/bash
if [ -f $USER_HOME/.VLprofile ]; then source $USER_HOME/.VLprofile; fi

#export PYTHONPATH=/home/ibsim/VirtualLab/Scripts:/home/ibsim/VirtualLab/Scripts/Common:/home/ibsim/VirtualLab/Scripts/Experiments/HIVE:${PYTHONPATH}

command=$1 
VL_hostname=$2 # name of the host which VirtualLab was initially launched on
VL_port=$3 #  the port on which VirtualLab was initially launched
shared_dir=$4 # needed when running on multiple nodes

# slightly complex method of getting the host name, but environment variables 
# are not updated when MPI is launched, so we can't use $HOST/$HOSTNAME.
# Instead we use hostname protocol in VirtualLab executable. 
temp_dir=$(mktemp -d)
host_file=$temp_dir/host
VirtualLab hostname $host_file
host=`cat $host_file`

if [ $host = $VL_hostname ] ; then
    # task is being run on the same node as what the VirtualLab run file is run on,
    # so we use the same tcp_port already setup
    $command $VL_port
else
    # task is run on a diffrent node so must set up a new tcp port to run on
    # create server on new node and write port number to port_file
    port_file=$temp_dir/port

    VirtualLab server_start $port_file $shared_dir & # Create server and send to background
    while true ; do
        if [ -f $port_file ] ; then 
            port=`cat $port_file`
            break 
        fi  
        done
    $command $port
    VirtualLab server_kill $port # kill server created on new node

fi


rm -r $temp_dir
