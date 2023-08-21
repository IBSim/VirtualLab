
Containers
===========

**VirtualLab** uses software and codes placed in containers to perform analysis.

If you're unfamiliar with containers, here's a quick overview from `opensource.com <https://opensource.com/resources/what-are-linux-containers>`_\ :footcite:`containers`:

    *"Containers, in short, contain applications in a way that keep them isolated from the host system that they run on. Containers allow a developer to package up an application with all of the parts it needs, such as libraries and other dependencies, and ship it all out as one package. And they are designed to make it easier to provide a consistent experience as developers and system administrators move code from development environments into production in a fast and replicable way.*

    *In a way, containers behave like a virtual machine. To the outside world, they can look like their own complete system. But unlike a virtual machine, rather than creating a whole virtual operating system, containers don't need to replicate an entire operating system, only the individual components they need in order to operate. This gives a significant performance boost and reduces the size of the application. They also operate much faster, as unlike traditional virtualization the process is essentially running natively on its host, just with an additional layer of protection around it."*

We have chosen containers as the main way of distributing **VirtualLab** for a number of reasons:

* We, the developers, take on the effort to ensure that all software dependencies are met meaning that the users can focus on getting up and running as quickly as possible.
* The portability of containers means that, whether working on a laptop or a HPC cluster, a container pull (or download) is all that's required and users' workflows can be easily moved from machine to machine when scaling up to a larger resource is required.
* The small impact on performance is far outweighed by the benefits of easy installation compared with a local installation.
* Containers offer superior performance compared with virtual machines and can make use of hardware acceleration with GPUs.
* Containers allow us to install external modules each with their own dependencies isolated from one another.

Currently the only containerisation tool supported by **VirtualLab** is `Apptainer <https://apptainer.org/>`_ . This is only available on Linux, hence Linux being the only officially supported OS. 

.. note::
    Apptainer's website does contain instructions for using it Windows and MacOs. However, this through a virtual machine which prevents the use of GPUs for modules that support them. It also has a negative impact on performance as such we don't recommend using Apptainer on non Linux systems. 

Manager
*******

The central component of the **VirtualLab** platform is the Manager container. This uses the VirtualLab python package to execute the steps of the RunFile, passing jobs to other containers via a server which runs on your local machine. 

.. image:: https://gitlab.com/ibsim/media/-/raw/master/images/VirtualLab/VL_Worflowpng_v2.png
  :width: 800
  :alt: Diagram of VirtualLab container setup
  :align: center


SalomeMeca
***********

Container which includes the 2019 version of `SalomeMeca <https://code-aster.org/V2/spip.php?article303>`. **SalomeMeca** is the pre and post-processing software `SALOME <https://www.salome-platform.org/>`_ with the Finite Element (FE) solver `Code_Aster <https://code-aster.org/V2/spip.php?article272>` integrated within it.

This container also includes the electro-magnetic FE solver `ERMES <http://tts.cimne.com/ermes/index.html>`.

Cad2Vox
********

Contains the Cad2Vox package for mesh voxelisation.

CIL
****

Contains the CIL package for reconstruction of CT data. 

GVXR
*****









References
**********
.. footbibliography::