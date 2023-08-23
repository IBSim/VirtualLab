.. role:: bash(code)
   :language: bash
	      
Installation & configuration
============================

The **VirtualLab** platform has been designed so that only a small number of prerequisites are needed for its use. `Containers <containers.html>`_ are used to house various codes and software, meaning that a containerisation tool is required, with `Apptainer <https://apptainer.org/>`_ the chosen tool. The **VirtualLab** python package only requires gitpython (and therefore git), and can be used with either native python (including pip) or conda. 

**VirtualLab** Supports the following operating systems:

.. list-table:: Supported OS's
  :widths: 25 25 50
  :header-rows: 1
  
  * - Operating System
    - Version
    - Notes
  * - Linux
    - Mint 19/Ubuntu 20.04+
    - Any reasonably modern distro should work. **VirtualLab** has been tested on various desktops and laptops running Ubuntu 20.04 LTS and a supercomputer running Redhat Linux enterprise 9. However, as with all things Linux results may vary on other distros.
  
As Apptainer is only available on Linux this is currently the only officially supported OS. 

.. note::
    Apptainer's website does contain instructions for using it Windows and MacOs. However, this through a virtual machine which prevents the use of GPUs for modules that support them. It also has a negative impact on performance as such we don't recommend using Apptainer on non Linux systems. 

Quick Installation
******************

You may run the following 'one line' command on a fresh installation of a supported Linux distro to install **VirtualLab** and the containerisation tool. The only requirement for this installation is that you have either python installed (including pip), or conda.

.. warning::
  If you are not using a fresh OS it is highly recommended that you read the rest of this page before trying this method, to better understand how the installation is carried out in case you have pre-installed dependencies which might brake.

  This 'one line' commant has only been tested on Ubuntu 20.04 LTS.

Terminal::

    wget https://gitlab.com/ibsim/virtuallab/-/raw/master/Scripts/Install/Host/Install_main.sh && \
    chmod 755 Install_main.sh && \
    ./Install_main.sh -y  && \
    rm Install_main.sh && \
    source ~/.VLprofile

Running the above will install both git and apptainer along with the VirtualLab python package using standard python.

.. note::
  Install_main.sh contains a couple of sudo commands, therefore you will be promted to enter your password.

The :code:`-y` in the third line of the above signals to skip the installation dialogue and install **VirtualLab** in the default location :file:`/home/$USER/VirtualLab`. This flag can be removed if youd require a non-standard install. 

.. note::
  If youd like to install **VirtualLab** using conda or to get the latest development version see `here <install.html#standard-install>`_.

To test out the install follow the steps outlined `here <install.html#testing>`_.

Standard install
*****************

To use the install/update script you will need to install git. This can be easily done by either following the instructions on `git's website <https://git-scm.com/download/linux>`_ or, on Ubuntu based distros, you can run the following in a terminal.

:bash:`sudo apt install git`

**VirtualLab** is primarily command line only so you will need to run the following commands in a terminal to install the **VirtualLab** python package  ::
  
      wget https://gitlab.com/ibsim/virtuallab/-/raw/master/Scripts/Install/Host/Install_VirtualLab.sh && \
      chmod 755 Install_VirtualLab.sh && \
      ./Install_VirtualLab.sh  && \
      rm Install_VirtualLab.sh && \
      source ~/.VLprofile

The installer will then take you through a series of menus and download the latest version of the code.

.. note:: 
  
  If you'd prefer to use conda instead of native python you will need to add :code:`-I conda` to the third line, e.g. ::

    ./Install_VirtualLab.sh -I conda

  This will create an environment named VirtualLab.

.. note::
  The above will install the most recent version of **VirtualLab**. The latest development version can be installed from the dev branch using the following::

      BRANCH=dev
      wget https://gitlab.com/ibsim/virtuallab/-/raw/${BRANCH}/Scripts/Install/Host/Install_VirtualLab.sh && \
      chmod 755 Install_VirtualLab.sh && \
      ./Install_VirtualLab.sh -B $BRANCH  && \
      rm Install_VirtualLab.sh && \
      source ~/.VLprofile

  where the -B flag indicates the branch from which **VirtualLab** will be installed. 

Next you will need to install Apptainer. This can either be installed using the following::

    wget https://gitlab.com/ibsim/virtuallab/-/raw/master/Scripts/Install/Host/Install_Apptainer-bin.sh && \
    chmod 755 Install_Apptainer-bin.sh && \
    sudo ./Install_Apptainer-bin.sh -y  && \
    rm Install_Apptainer-bin.sh

or by following the most up-to-date instructions from their website:

* `Quick start <https://apptainer.org/docs/user/main/quick_start.html>`_
* `Install Apptainer <https://apptainer.org/docs/admin/main/installation.html>`_

At this point the **VirtualLab** package will have been installed, however none of the containers it requires have yet been downloaded. These will be installed as and when they are needed for the analysis in question. 

The size of these containers can be quite large. As standard, these containers will be saved to a directory named 'Containers' in the VirtualLab directory. If you'd prefer these containers be saved elsewhere, this can be changed in :file:`VLconfig.py` file in the VirtualLab directory, see `code configuration <../structure.html#code-configuration>`_ for more details. 

To test out the install follow the steps outlined `here <install.html#testing>`_.

Testing
*******

To test out that the installation has worked as expected run the following command

:bash:`VirtualLab --test`



This will download **VirtualLab**'s `containers.html#manager`_ container along with a small test container to make sure things are set up correctly. It also spits out a randomly selected programming joke as a nice whimsical bonus.

For more on how to use **VirtualLab** we recommend working through the `Tutorials <examples/index.html>`_ section.

MPI
***

**VirtualLab** is able to perform analysis on multi-node systems as well as personal computers. For this MPI is required, and needs to be compatible with the MPI installed within **VirtualLab**'s `containers.html#manager`_ container, which is `MPICH <https://www.mpich.org/>`_. To install MPICH run the following command ::

  sudo apt install mpich

To test out that **VirtualLab** is compatible with MPI run the following ::

  VirtualLab -f RunFiles/MPI_test.py

You should see an output similar to this (order will differ) ::

  Hello! I'm rank 0 from 5 running in total...
  Hello! I'm rank 2 from 5 running in total...
  Hello! I'm rank 1 from 5 running in total...
  Hello! I'm rank 4 from 5 running in total...
  Hello! I'm rank 3 from 5 running in total...


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

