Installation
============

To use **VirtualLab** there are a number of prerequisites. **VirtualLab** has only been tested to work on the operating system (OS) `Ubuntu 18.04 LTS (Bionic Beaver) <https://releases.ubuntu.com/18.04/>`_. Your mileage may vary on other OSs.

Non-interactive Installation
****************************

The easiest way to download & install **VirtualLab** and its dependencies in a conda environment on **Ubuntu** is by running the following command in a terminal::

    cd ~ && wget -O Install_VirtualLab.sh https://gitlab.com/ibsim/virtuallab/-/raw/master/Scripts/Install/Install_VirtualLab.sh?inline=false && chmod 755 Install_VirtualLab.sh && sudo ~/./Install_VirtualLab.sh -P c -S y -y && source ~/.bashrc 


Usage of 'Install_VirtualLab.sh':
  Install_VirtualLab.sh [-d <path>] [-P {y/c/n}] [-S \"{y/n} <path>\"]

Options:
   | :bash:`-d <path>` Specify the installation path for **VirtualLab**.
   | :bash:`-P y` Install python3 using system python installation.
   | :bash:`-P c` Install python3 using conda environment.
   | :bash:`-P n` Do not install python.
   | :bash:`-S \"y <path>\"` Install **Salome-Meca** at *<path>* location.
   | :bash:`-S y` Install **Salome-Meca** at default location *'/opt/SalomeMeca'*.
   | :bash:`-S n` Do not install **Salome-Meca**.
   | :bash:`-y` Skip install confirmation dialogue.

* The default behaviour (with no flags) is to not install any version of python or **Salome-Meca** (which includes **Code_Aster**).
* If creating a conda environment, it will be named the same as the installation directory for **VirtualLab** (which is 'VirtualLab' by default).
* The default installation locations are:

  + **VirtualLab** in the user's home directory :bash:`$HOME`.
  + **Salome-Meca** in *'/opt/SalomeMeca'*.
  + python/conda in the conventional locations.

If you have a pre-existing installation of any of the components the script will attempt to detect this and only update necessary components in order to ensure that dependencies are met.

Manual Installation
*******************

If you choose to perform the installation manually, you will need to install each of the various components and ensure that **VirtualLab** and **Salome-Meca** are added to your system :bash:`$PATH`. Additionally, **VirtualLab** will need to be added to your :bash:`$PYTHONPATH`.

The python package requirements are found here: **LINK IN DOCS**.

To complete the installation you will need to run *'SetupConfig.sh'*. If you have used any non-default installation options, you will first need to modify *'VLconfig_DEFAULT.sh'*. *'SetupConfig.sh'* will attempt to locate non-default options, but manually modifying *'VLconfig_DEFAULT.sh'* is a failsafe way of ensuring configuration completes succesfully.

Then run the following command in the location where you have installed **VirtualLab**::

  ./SetupConfig.sh

Usage of 'SetupConfig.sh':
  SetupConfig.sh [ -f "$FNAME" ]

Options:
   | :bash:`-f "$FNAME"` Where "$FNAME" is the name of the config options file (e.g. *'VLconfig_DEFAULT.sh'*).

 * The default behaviour is to setup using VLconfig_DEFAULT.sh.
 * If you change any of the config options you will need to re-run *'SetupConfig.sh'* for changes to be applied.

Alternative Options
*******************

In future, we hope to offer **VirtualLab** to be downloaded as a virtual machine (VM) or container to facilitate portability.
