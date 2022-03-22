.. role:: bash(code)
   :language: bash
	      
Installation
============

To use **VirtualLab** there are a number of prerequisites. **VirtualLab** has only been tested to work on the operating system (OS) `Ubuntu 18.04 LTS (Bionic Beaver) <https://releases.ubuntu.com/18.04/>`_. Your mileage may vary on other OS's.

Containers
**********

The recommended way of using **VirtualLab** on your system is via a container. It has been tested and works with both `Docker <https://www.docker.com/>`_ and `Singularity <https://sylabs.io/singularity/>`_.

If you're unfamiliar with containers, here's a quick overview from `opensource.com <https://opensource.com/resources/what-are-linux-containers>`_ :cite:`containers`:

    *"Containers, in short, contain applications in a way that keep them isolated from the host system that they run on. Containers allow a developer to package up an application with all of the parts it needs, such as libraries and other dependencies, and ship it all out as one package. And they are designed to make it easier to provide a consistent experience as developers and system administrators move code from development environments into production in a fast and replicable way.*

    *In a way, containers behave like a virtual machine. To the outside world, they can look like their own complete system. But unlike a virtual machine, rather than creating a whole virtual operating system, containers don't need to replicate an entire operating system, only the individual components they need in order to operate. This gives a significant performance boost and reduces the size of the application. They also operate much faster, as unlike traditional virtualization the process is essentially running natively on its host, just with an additional layer of protection around it."*

We have chosen containers as the main way of distributing **VirtualLab** for a number of reasons:

* We, the developers, take on the effort to ensure that all software dependencies are met meaning that the users can focus on getting up and running as quickly as possible.
* The portability of containers means that, whether working on a laptop or a HPC cluster, a container pull (or download) is all that's required and users' workflows can be easily moved from machine to machine when scaling up to a larger resource is required.
* The small impact on performance is far outweighed by the benefits of easy installation compared with a local installation.
* Containers offer superior performance compared with virtual machines and can make use of hardware acceleration with GPUs.

To use **VirtualLab** with a container you must first install your platform of choice. Is it suggested that you follow the most up-to-date instructions from their websites:

* `Install Docker <https://docs.docker.com/get-docker/>`_
* `Install Singularity v3.6 <https://sylabs.io/guides/3.6/user-guide/quick_start.html#quick-installation-steps>`_

Then run the appropriate command below:

Docker::

    sudo docker pull ibsim/virtuallab:latest

Singularity::

    singularity pull VirtualLab.sif docker://ibsim/virtuallab:latest

Docker supports Windows, Mac and Linux whereas Singularity primarily only supports Linux. The **VirtualLab** container has only been tested with these platforms on `Ubuntu 18.04 LTS (Bionic Beaver) <https://releases.ubuntu.com/18.04/>`_ and with Singularity on the `Supercomputing Wales <https://www.supercomputing.wales/>`_ HPC resource.

The following commands will work with a fresh installation of `Ubuntu 18.04 LTS (Bionic Beaver) <https://releases.ubuntu.com/18.04/>`_ to install your container platform of choice and pull **VirtualLab**.

Docker::

    cd ~ && wget -O https://gitlab.com/ibsim/virtuallab/-/raw/master/Scripts/Install/Install_Docker.sh?inline=false && chmod 755 Install_Docker.sh && sudo ~/./Install_Docker.sh && source ~/.bashrc && sudo docker pull ibsim/virtuallab:latest

Singularity::

    cd ~ && wget -O https://gitlab.com/ibsim/virtuallab/-/raw/master/Scripts/Install/Install_Singularity.sh?inline=false && chmod 755 Install_Singularity.sh && sudo ~/./Install_Singularity.sh && source ~/.bashrc && singularity pull VirtualLab.sif docker://ibsim/virtuallab:latest

+--------------------------------------------------------------------------+----------------+---------------+-------------+---------+
| Image name                                                               | Build          | Description   | Salome-Meca | ERMES   |
|                                                                          | Status         |               | Version     | Version |
+==========================================================================+================+===============+=============+=========+
| `docker://ibsim/virtuallab <https://hub.docker.com/r/ibsim/virtuallab>`_ | |build-status| | alpha release | 2019.0.3    | 12.5    |
+--------------------------------------------------------------------------+----------------+---------------+-------------+---------+

.. |build-status| image:: https://img.shields.io/docker/cloud/build/ibsim/virtuallab

Virtual Machines
****************

The next most straightforward method to use **VirtualLab** is to download the preprepared virtual machine (VM) image. This is a complete Ubuntu OS desktop environment with all the necessary software and dependencies pre-installed that can be run on your current computing setup. There are three steps to this method:

#. Download and install the virtual machine platform `VirtualBox <https://www.virtualbox.org/wiki/Downloads>`_ following the appropriate instructions for your OS (Windows, Mac or Linux).
#. Download the |VM_link|.
#. Load the image :file:`File > Import Appliance...`
#. (Optional) Amend the VM settings for your specific hardware e.g. increase allocation of CPU cores or RAM.

.. |VM_link| raw:: html

   <a href="https://ibsim.co.uk/VirtualLab/downloads/VM.html" target="_blank">VirtualLab image</a>

These are the login details for the VM:

* username = ibsim
* password = ibsim

The limitation of VMs is that they cannot access your GPU for graphical acceleration and there will be a non-negligible impact to performance. However, this is a sufficiently smooth user experience for the majority of use-cases.

Non-interactive Installation
****************************

The easiest way to download & install **VirtualLab** and its dependencies in a conda environment on **Ubuntu** is by running the following command in a terminal::

    cd ~ && wget -O Install_VirtualLab.sh https://gitlab.com/ibsim/virtuallab/-/raw/master/Scripts/Install/Install_VirtualLab.sh?inline=false && chmod 755 Install_VirtualLab.sh && sudo ~/./Install_VirtualLab.sh -P c -S y -E y -y && source ~/.bashrc


Usage of 'Install_VirtualLab.sh':
  Install_VirtualLab.sh [-d <path>] [-P {y/c/n}] [-S \"{y/n} <path>\"] [-E {y/n}]

Options:
   | :bash:`-d <path>` Specify the installation path for **VirtualLab**.
   | :bash:`-P y` Install python3 using system python installation.
   | :bash:`-P c` Install python3 using conda environment.
   | :bash:`-P n` Do not install python.
   | :bash:`-S y <path>` Install **Salome-Meca** at *<path>* location.
   | :bash:`-S y` Install **Salome-Meca** at default location *'/opt/SalomeMeca'*.
   | :bash:`-S n` Do not install **Salome-Meca**.
   | :bash:`-E y` Install **ERMES** at default location *'/opt/ERMES`*
   | :bash:`-E n` Do not install **ERMES**
   | :bash:`-C y` Install **Cad2Vox**
   | :bash:`-C n` Do not install **Cad2Vox**
   | :bash:`-y` Skip install confirmation dialogue.

* The default behaviour (with no flags) is to not install any version of python, **Salome-Meca** (which includes **Code_Aster**),**ERMES**, or **Cad2Vox**.
* If creating a conda environment, it will be named the same as the installation directory for **VirtualLab** (which is 'VirtualLab' by default).
* The default installation locations are:

  + **VirtualLab** in the user's home directory :bash:`$HOME`.
  + **Salome-Meca** in *'/opt/SalomeMeca'*.
  + **ERMES** in *'/opt/ERMES`*.
  + python/conda in the conventional locations.

If you have a pre-existing installation of any of the components the script will attempt to detect this and only update necessary components in order to ensure that dependencies are met.

Manual Installation
*******************

If you choose to perform the installation manually, you will need to install each of the various components and ensure that **VirtualLab**, **Salome-Meca** and **ERMES*** are added to your system :bash:`$PATH`. Additionally, **VirtualLab** will need to be added to your :bash:`$PYTHONPATH`.

The python package requirements are found at the code's `git repository <https://gitlab.com/ibsim/virtuallab/-/raw/master/requirements.txt>`_.

To complete the installation you will need to run *'SetupConfig.sh'*. If you have used any non-default installation options, you will first need to modify *'VLconfig_DEFAULT.sh'*. *'SetupConfig.sh'* will attempt to locate non-default options, but manually modifying *'VLconfig_DEFAULT.sh'* is a fail-safe way of ensuring configuration completes successfully.

Then run the following command in the location where you have installed **VirtualLab**::

  ./SetupConfig.sh

Usage of 'SetupConfig.sh':
  SetupConfig.sh [ -f "$FNAME" ]

Options:
   | :bash:`-f "$FNAME"` Where "$FNAME" is the name of the config options file (e.g. *'VLconfig_DEFAULT.sh'*).

 * The default behaviour is to setup using VLconfig_DEFAULT.sh.
 * If you change any of the config options you will need to re-run *'SetupConfig.sh'* for changes to be applied.

References
**********

.. bibliography:: refs.bib
   :style: plain
   :filter: docname in docnames
