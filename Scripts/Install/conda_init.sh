#!/bin/bash
#USER_HOME=$(eval echo ~${SUDO_USER})
#if [ -f $USER_HOME/.VLprofile ]; then source $USER_HOME/.VLprofile; fi

#echo $USER_HOME
#sudo -s -u ${SUDO_USER} eval '"$('$USER_HOME'/anaconda3/bin/conda shell.bash hook)"'
#sudo -s -u ubuntu eval "$(/home/ubuntu/anaconda3/bin/conda shell.bash hook)"
#sudo -s -u ubuntu eval "$(conda init)"
#sudo -s -u ${SUDO_USER} conda init

#eval "$($USER_HOME/anaconda3/bin/conda shell.bash hook)"
#conda init

eval "$($HOME/anaconda3/bin/conda shell.bash hook)"
conda init
