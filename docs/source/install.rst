.. role:: bash(code)
   :language: bash
	      
Installation
============

To use **VirtualLab** there are very few prerequisites. To use VirtualLab only Apptainer is strictly required. However git is required for the installation.

**VirtualLab** Supports the following operating systems:

.. list-table:: Supported OS's
  :widths: 25 25 50
  :header-rows: 1
  
  * - Operating System
    - Version
    - Notes
  * - Linux
    - Mint 19/Ubuntu 18.04+
    - Any reasonably modern distro should work. **VirtualLab** has been tested on various desktops and laptops running ubuntu 18.04 and 20.04 LTS and a supercomputer running Redhat Linux enterprise 9. However, as with all things Linux results may vary on other distros [1]_.
  
.. [1] Note: Builds are made with pyinstaller which can be downloaded `here <https://github.com/pyinstaller/pyinstaller>`_. For linux this can generate builds for Arm64 (Raspberry Pi) and IBM PowerPc. These aren't officially supported, due to lack of demand/resources but it's there should the need arise.

Quick Installation
******************

You may run the following 'one line' command on a fresh installation of a supported Linux distro to install **VirtualLab** and the containerisation tool.

.. warning::
  If you are not using a fresh OS it is highly recommended that you read the rest of this page before trying this method, to better understand how the installation is carried out in case you have pre-installed dependencies which might brake.

Terminal::

    cd && wget https://gitlab.com/ibsim/virtuallab/-/raw/dev/Scripts/Install/Install_VLplus.sh && chmod 755 Install_VLplus.sh && sudo ./Install_VLplus.sh -B d -y && source ~/.VLprofile && rm Install_VLplus.sh

Containers
**********

The only other prerequisite for using **VirtualLab** on your system is a containerisation tool. We currently only support `Apptainer <https://apptainer.org/>`_ (currently this is Linux only, hence Linux being the only officially supported OS). [3]_ 

If you're unfamiliar with containers, here's a quick overview from `opensource.com <https://opensource.com/resources/what-are-linux-containers>`_\ :footcite:`containers`:

    *"Containers, in short, contain applications in a way that keep them isolated from the host system that they run on. Containers allow a developer to package up an application with all of the parts it needs, such as libraries and other dependencies, and ship it all out as one package. And they are designed to make it easier to provide a consistent experience as developers and system administrators move code from development environments into production in a fast and replicable way.*

    *In a way, containers behave like a virtual machine. To the outside world, they can look like their own complete system. But unlike a virtual machine, rather than creating a whole virtual operating system, containers don't need to replicate an entire operating system, only the individual components they need in order to operate. This gives a significant performance boost and reduces the size of the application. They also operate much faster, as unlike traditional virtualization the process is essentially running natively on its host, just with an additional layer of protection around it."*

We have chosen containers as the main way of distributing **VirtualLab** for a number of reasons:

* We, the developers, take on the effort to ensure that all software dependencies are met meaning that the users can focus on getting up and running as quickly as possible.
* The portability of containers means that, whether working on a laptop or a HPC cluster, a container pull (or download) is all that's required and users' workflows can be easily moved from machine to machine when scaling up to a larger resource is required.
* The small impact on performance is far outweighed by the benefits of easy installation compared with a local installation.
* Containers offer superior performance compared with virtual machines and can make use of hardware acceleration with GPUs.
* Containers allow us to install external modules each with their own dependencies isolated from one another.

For **VirtualLab** we use a number of different containers (modules) that are co-ordinated by a Manager container with inter-container communication being handled by a small server application that runs on your local machine (we will go into more details on exactly how this all works later).

.. image:: https://gitlab.com/ibsim/media/-/raw/master/images/VirtualLab/VL_Worflowpng.png
  :width: 400
  :alt: Diagram of VirtualLab container setup
  :align: center

To use **VirtualLab** you must first install Apptainer. It is suggested that you follow the most up-to-date instructions from their website:

* `Quick start <https://apptainer.org/docs/user/main/quick_start.html>`_
* `Install Apptainer <https://apptainer.org/docs/admin/main/installation.html>`_

.. [3] Apptainer's website does contain instructions for using it Windows and MacOs. However, this through a virtual machine which prevents the use of GPUs for modules that support them. It also has a negative impact on performance as such we don't recommend using Apptainer on non Linux systems. 

Installation with the install script:
*************************************

To use the install/update script you will need to install git. This can be easily done by either following the instructions on `git's website <https://git-scm.com/download/linux>`_ or, on Ubuntu based distros, you can run the following in a terminal.

:bash:`sudo apt install git`

Once you have git and Apptainer installed you can download the automated install/update `script <https://gitlab.com/ibsim/virtuallab_bin/-/raw/main/Install_VirtualLab?inline=false>`_:

Both the Installer and **VirtualLab** itself are primarily command line only so you will need to run the following commands in a terminal.

:bash:`chmod +x Install_VirtualLab`

:bash:`./Install_VirtualLab` 

The installer will then take you through a series of menus and download the latest version of the code as well as pulling the latest **VirtualLab** Manager container from Dockerhub (converting it to an apptainer container).

.. note:: You may see lots of warning messages appear on screen during the install, similar to: :bash:`warn rootless {path/to/file} ignoring (usually) harmless EPERM on setxattr`. As the messages suggests these are harmless and just a bi-product of building containers from sif files without root privileges on Linux. Thus, as long as you get a "build complete" message at the end they can be safely ignored.

We note at this stage that only the 'Server' and 'Manager' have been downloaded. The remaining modules are not immediately installed but instead will be downloaded and installed dynamically when used for the first time. This means that the first run of any module will take significantly longer because it has to download and install the required files. This is an intentional trade off to save disk space because it means you only have installed the exact tools you need/use.

The **VirtualLab** executable can then be found in the bin directory inside the **VirtualLab** install directory (you may want to add this to your system path).

.. note:: Unless you changed it during the install the default install directory is :bash:`/home/$USER/VirtualLab` where $USER is your username.

We recommend you run a quick test to ensure everything is working this can be done with the following command:

:bash:`VirtualLab --test`

The :bash:`--test` option downloads a minimal test container and runs a series of tests to check everything is working. It also spits out a randomly selected programming joke as a nice whimsical bonus. For more on how to use **VirtualLab** we recommend the `Tutorials <examples/index.html>`_ section.

.. warning:: **GlibC issues with Ubuntu 22.04+**
  
  We note, at this stage, that there is a known bug with Salome-Meca Running in VirtualLab with Ubuntu 22.04, along with some newer versions of Fedora. 
  If you are using these you may find you get an error containing something similar to the following:
  ``version `GLIBC_2.34' not found (required by /.singularity.d/libs/libGLX.so.0)``
  
  The issue is a bug in the way that the ``--nv`` flag loads nvidia libraries. The short version is that the ``--nv`` flag isn't very sophisticated when it comes to libraries. It looks for a list of library files on the host which is defined in ``nvliblist.conf``. 
  The issue is that the latest version(s) of Ubuntu are compiled against a newer version of libGLX than is included within the Salome container. This causes problems in Apptainer.

  To fix this you have two options. Firstly, you can use the ``-N`` option to turn off the nvidia libraries. The drawback to this is that you will be running in 'software rendering mode' and thus you will not benefit from any GPU acceleration.

  The second option is to use the following workaround.

  1. Search for a file named ``nvliblist.conf`` in your installation. It should be under your Apptainer installation directory. By default this is under ``/etc/apptainer``.
  2. Make a back-up of this file ``mv nvliblist.conf nvliblist.conf.bak``.
  3. Open the file ``nvliblist.conf`` using a text editor.
  4. Delete all of the following lines that appear ``libGLX.so``, ``libGLX.so.0``, ``libglx.so``, ``libglx.so.0`` and ``libGLdispatch.so``. Note, depending on you exact system, the file may not contain all of them.

  Try running the Salome container again, it should work this time.

  Reference: https: //github.com/apptainer/apptainer/issues/598
  
  One caveat with this workaround, however, is that involves messing with configs that apply system wide. As such, it may have unintended side-effects with other software/containers that use Apptainer. Our team have not yet reported any issues. 
  However, this does not mean they do not exist. Therefore, we cannot 100% guarantee you won't have any issues. This is also the reason we recommend backing up your original config in step 2, just in case. Also, for future 
  reference, these fixes where applied to ubuntu 22.04 with Apptainer version 1.0.5. Your millage may vary with future updates.

Installation from source code
*****************************

If you choose to perform the installation manually, in addition to Apptainer you will need both `git <https://git-scm.com/download/linux>`_, `python <https://www.python.org/>`_ version 3.9+ and optionally the pip package `pyinstaller <https://pyinstaller.org/en/stable/>`_. 

First, you will need to clone our git repository with:
:bash:`git clone https://gitlab.com/ibsim/virtuallab.git`

Next, you need to download the latest version of the manager container from dockerhub. To do this run :bash:`apptainer build VL_Manager.sif docker://ibsim/virtuallab:latest` then place the generated VLManager.sif file into the Containers directory of the **VirtualLab** repository which you cloned in the previous step.

The next step is to generate an executable. The original script the executable is based on is VL_server.py. So from here you have essentially 2 options:

1. Run the script directly with :bash:`python3 VL_server.py --test`
2. Build an executable yourself using pyinstaller by running :bash:`pyinstaller -n VirtualLab -F VL_server.py`

.. note:: As mentioned previously, all the other container modules get downloaded automatically the first time they are used. However, regardless of your container choice they are all hosted on dockerhub under `ibsim <https://hub.docker.com/u/ibsim>`_. You could always pull/build them from there if desired. Alternatively, the dockerfiles used to create the containers can be found in a separate github `repo <https://github.com/IBSim/VirtualLab>`_ that is itself linked to Dockerhub.

The final step is to add **VirtualLab** to the system path and set the VL_DIR environment variable to tell **VirtualLab** where the code is installed.

To do this run the following commands:
:bash:`export VL_DIR=Path/to/repo`
:bash:`export PATH=$PATH:{Path/to/repo}/bin`

.. note:: You may want to automate this by adding these lines to ~/.bashrc, ~/.zshrc or similar.

References
**********
.. footbibliography::
