.. role:: bash(code)
   :language: bash
	      
Installation
============

To use **VirtualLab** there are very few prerequisites which depend on you Operating system (OS). **VirtualLab** Supports the following operating systems:

.. list-table:: Supported OS's
  :widths: 25 25 50
  :header-rows: 1
  
  * - Operating System
    - Version
    - Notes
  * - Windows (32/64bit)
    - 10/11
    - x86 only [1]_
  * - MacOs
    - 10.15 or newer
    -  We support x86_64 (intel) or arm64 (M1/Apple silicon) via universal binaries.
  * - Linux
    - Mint 19/Ubuntu 18.04+
    - Any reasonably modern distro should work. We have tested on various desktops and laptops running ubuntu 18.04 and 22.04 LTS and our supercomputer running Redhat Linux enterprise 9. However, as with all things Linux results may vary on other distros [2]_. 

.. [1] If you don't know what this means it likely does not apply to you (does anyone actually use Windows on Arm?).
.. [2] Note: builds are made with pyinstaller which can be downloaded `here <https://github.com/pyinstaller/pyinstaller>`_. For linux this can generate builds for Arm64 (Raspberry Pi) and IBM PowerPc. We don't officially support this, due to lack of demand/resources but it's there should the need arise.


Containers
**********

The only other prerequisite for using **VirtualLab** on your system is a containerisation tool. **VirtualLab** can work with either `Docker <https://www.docker.com/>`_ or `Apptainer <https://apptainer.org/>`_ (currently Linux only). [3]_

If you're unfamiliar with containers, here's a quick overview from `opensource.com <https://opensource.com/resources/what-are-linux-containers>`_ :cite:`containers`:

    *"Containers, in short, contain applications in a way that keep them isolated from the host system that they run on. Containers allow a developer to package up an application with all of the parts it needs, such as libraries and other dependencies, and ship it all out as one package. And they are designed to make it easier to provide a consistent experience as developers and system administrators move code from development environments into production in a fast and replicable way.*

    *In a way, containers behave like a virtual machine. To the outside world, they can look like their own complete system. But unlike a virtual machine, rather than creating a whole virtual operating system, containers don't need to replicate an entire operating system, only the individual components they need in order to operate. This gives a significant performance boost and reduces the size of the application. They also operate much faster, as unlike traditional virtualization the process is essentially running natively on its host, just with an additional layer of protection around it."*

We have chosen containers as the main way of distributing **VirtualLab** for a number of reasons:

* We, the developers, take on the effort to ensure that all software dependencies are met meaning that the users can focus on getting up and running as quickly as possible.
* The portability of containers means that, whether working on a laptop or a HPC cluster, a container pull (or download) is all that's required and users' workflows can be easily moved from machine to machine when scaling up to a larger resource is required.
* The small impact on performance is far outweighed by the benefits of easy installation compared with a local installation.
* Containers offer superior performance compared with virtual machines and can make use of hardware acceleration with GPUs.
* Containers allow us to install external modules each with there own dependencies isolated from one another.

For **VirtualLab** we use a number of different containers (modules) that are co-ordinated by a Manager container with inter-container communication being handled by a small server application that runs on you local machine (we will go into more details on exactly how this all works later.)

.. image:: path/filename.png
  :width: 400
  :alt: Diagram of VirtualLab container setup (add me)

For a full technical deep dive into how all this works under the hood check out the section on Container Paradigm.

To use **VirtualLab** you must first install your container platform of choice. Is it suggested that you follow the most up-to-date instructions from their websites:

* `Install Docker <https://docs.docker.com/get-docker/>`_
* `Install Apptainer <https://apptainer.org/docs/user/main/quick_start.html>`_

.. [3] Apptainer's website does contain instructions for using it Windows and MacOs. However, this through a virtual machine which prevents the use of GPUs for modules that support them. It also has a negative impact on performance as such we don't recommend using Apptainer on non Linux systems. 

A note for Windows users:
*************************

Some VirtualLab modules have graphical user interfaces for running in "interactive" mode. However due to the difference in the way windows and Linux handle graphical applications you will need an additional piece of software called an xserver.

This optional, since all modules are required to be be able to run without a GUI in "headless" mode (see Tutorials for more details). However, `VcXsrv <https://sourceforge.net/projects/vcxsrv/>`_ is a common choice and can be installed using it's default settings. You will however need to remember to start the xserver program before running VirtualLab in interactive mode.

Installation with the install script:
*************************************

Once you have either Docker or Apptainer installed you can download the automated install/update script for you OS of choice links to which can be found:

* Windows
* Mac 
* Linux

Both the Installer and VirtualLab itself are primarily command line only so you will need to run the following command in a terminal.

Linux/MacOs:
:bash:`./vlabinstall` 

Windows:
:bash:`./vlabinstall.exe`

The installer will then take you though a series of menus and download the latest version of the code as well as pulling the latest VirtualLab Manager container from Dockerhub (converting it to an apptainer container if necessary).

Note: The various module are not immediately installed but instead will be downloaded and installed dynamically when used for the first time (this is intentional to save disk space as it means you only have installed the exact tools you need/use).

VirtualLab executable can then be found in the bin directory inside VirtualLab install directory (you may want to add this to your system path). Note: unless you changed it during the install the default install is :bash:`C:\\Program files\\VirtualLab` on windows or :bash:`/home/$USER/VirtualLab` on Mac/Linux where $USER is your username.

We recommend you run a quick test to ensure everything is working this can be done with the the following command:

Linux/MacOs:
:bash:`VirtualLab --test` 

Windows:
:bash:`VirtualLab.exe --test`

The --test option downloads a minimal test container and runs a series of tests to check everything is working. It also spits out a randomly selected programming joke as a nice whimsical bonus. For more on how to use VirtualLab we recommend the Tutorials section.


Installation from source code
*****************************

If you choose to perform the installation manually, in addition to Docker/Apptainer you will need both `git <https://git-scm.com/downloads>`_, `python <https://www.python.org/>`_ version 3.9+ and optionally the pip package `pyinstaller <https://pyinstaller.org/en/stable/>`_. 

First you will need clone our git repository with:
:bash:`git clone https://gitlab.com/ibsim/virtuallab.git`

Next you need to download the latest version of the manager container from dockerhub. To do this for Docker simply run :bash:`docker pull ibsim/virtuallab:latest`.

Whereas for Apptainer you need to run :bash:`singularity build VLManager.sif docker://ibsim/virtuallab:latest` then place the generated VLManager.sif file into the Containers directory of the repository.

The next step is to generate an executable. The original script the executable is based on is VL_server.py. So from here you have essentially 3 options:

1. use the pre-built VirtualLab executable in the bin directory
2. run the script directly with :bash:`python3 VL_server.py --test`
3. Build a new executable yourself using pyinstaller by running :bash:`pyinstaller -n VirtualLab -F VL_server.py`

.. note:: As mentioned previously all the other container modules get downloaded automatically the first time they are used. However, regardless of your container choice they are all hosted on dockerhub under `ibsim <https://hub.docker.com/search?q=ibsim>`_. So you can always pull/build them from there if desired. Alternatively the dockerfiles used to create the containers can be found in a separate github `repo <https://github.com/IBSim/VirtualLab>`_ that is itself linked to Dockerhub.


The final step is to add VirtualLab to the system path and set the VL_dir environment variable to tell VirtualLab where the code is installed.

On Linux/MacOs simply run:
:bash:`export VL_dir=Path/to/repo`
:bash:`export PATH=$PATH:Path/to/repo/bin`
Note: You may want to automate this by adding these lines to ~/.bashrc, ~/.zshrc or similar.

For Windows you will need to use the command prompt to set by running:
:bash:`setx VL_dir Path/to/repo`
:bash:`setx PATH "%PATH%;Path/to/repo/bin"`

Keeping VirtualLab up to date:
******************************
As we are no doubt aware updates are a necessary, if annoying part of modern software. Assuming you want bug fixes and shiny new features that is. Fortunately git and Dockerhub can, hopefully, make this a relatively painless experience.

If you are using Docker, modules should be automatically updated to latest version as soon as it uploaded to Dockerhub. 

References
**********
.. bibliography:: refs.bib
   :style: plain
   :filter: docname in docnames
