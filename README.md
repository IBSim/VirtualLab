# VirtualLab

Series of simulations to replicate laboratory experiments as part of a 'digital twin' initiative between Swansea University and the UK Atomic Energy Authority. Scripts are mostly in python to run EdF's simulation packages SALOME and Code_Aster.

To set up VirtualLab you must first run 'SetupConfig.sh' from the directory to which you have downloaded the source code. If you would like to change any of the default configuration you can do this from VLconfig_DEFAULT.sh.

It is possible to install VirtualLab and its dependencies by running the following command in the terminal.
`cd ~ && wget https://ibsim.co.uk/scripts/Install_VirtualLab.sh && chmod 755 Install_VirtualLab.sh && sudo ~/./Install_VirtualLab.sh -P c -S y -y`

See VirtualLab/docs for documentation.

   Copyright 2020 IBSim Group (c/o Llion Marc Evans)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
   