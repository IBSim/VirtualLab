# VirtualLab

Series of simulations to replicate laboratory experiments as part of a 'digital twin' initiative between Swansea University and the UK Atomic Energy Authority. Scripts are mostly in python to run EdF's simulation packages SalomeMeca and Code_Aster.

To set up VirtualLab you must first run 'SetupConfig.sh' from the directory to which you have downloaded the source code. If you would like to change any of the default configuration you can do this from VLconfig.sh.

It is possible to install VirtualLab and its dependencies by running the following command in the terminal.
`cd ~ && wget https://ibsim.co.uk/scripts/Install_VirtualLab.sh && chmod 755 Install_VirtualLab.sh && sudo ~/./Install_VirtualLab.sh -P c -S y`