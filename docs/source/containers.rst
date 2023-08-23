
Containers
===========

**VirtualLab** utilises software and codes placed in containers to perform analysis.

What are they?
***************

If you're unfamiliar with containers, here's a quick overview from `opensource.com <https://opensource.com/resources/what-are-linux-containers>`_\ :footcite:`containers`:

    *"Containers, in short, contain applications in a way that keep them isolated from the host system that they run on. Containers allow a developer to package up an application with all of the parts it needs, such as libraries and other dependencies, and ship it all out as one package. And they are designed to make it easier to provide a consistent experience as developers and system administrators move code from development environments into production in a fast and replicable way.*

    *In a way, containers behave like a virtual machine. To the outside world, they can look like their own complete system. But unlike a virtual machine, rather than creating a whole virtual operating system, containers don't need to replicate an entire operating system, only the individual components they need in order to operate. This gives a significant performance boost and reduces the size of the application. They also operate much faster, as unlike traditional virtualization the process is essentially running natively on its host, just with an additional layer of protection around it."*

Why we use them?
*****************

We have chosen containers as the main way of distributing **VirtualLab** for a number of reasons:

* We, the developers, take on the effort to ensure that all software dependencies are met meaning that the users can focus on getting up and running as quickly as possible.
* The portability of containers means that, whether working on a laptop or a HPC cluster, a container pull (or download) is all that's required and users' workflows can be easily moved from machine to machine when scaling up to a larger resource is required.
* The small impact on performance is far outweighed by the benefits of easy installation compared with a local installation.
* Containers offer superior performance compared with virtual machines and can make use of hardware acceleration with GPUs.
* Containers allow us to install external modules each with their own dependencies isolated from one another.


Available containers
**********************

Manager
#######

The central component of the **VirtualLab** platform is the 'Manager' container. This uses the VirtualLab python package to execute the steps of the RunFile, passing jobs to other containers via a server which runs on your local machine. 

.. image:: https://gitlab.com/ibsim/media/-/raw/master/images/VirtualLab/VL_Worflowpng_v2.png
  :width: 800
  :alt: Diagram of VirtualLab container setup
  :align: center

SalomeMeca
###########

Container which includes the 2019 version of `SalomeMeca <https://code-aster.org/V2/spip.php?article303>`. **SalomeMeca** is the pre and post-processing software `SALOME <https://www.salome-platform.org/>`_ with the Finite Element (FE) solver `Code_Aster <https://code-aster.org/V2/spip.php?article272>` integrated within it.

This container also includes the electro-magnetic FE solver `ERMES <http://tts.cimne.com/ermes/index.html>`.

Cad2Vox
########

Contains the Cad2Vox package for mesh voxelisation.

CIL
####

Contains the CIL package for reconstruction of CT data. 

Container build status
***********************

The below table gives an overview of the containers which are, in some way, linked with **VirtualLab** and their current build status.

+----------------+--------------------------------------------------------------------------------------------+-------------------+---------------+-------------+
| Image Name     | Docker Pull                                                                                | Build             | Software      | Version     |
|                |                                                                                            | Status            |               |             |
+----------------+--------------------------------------------------------------------------------------------+-------------------+---------------+-------------+
| vl_manager     | `docker://ibsim/virtuallab <https://hub.docker.com/r/ibsim/virtuallab>`_                   | |build-status_vl| | VirtualLab    | 22.0.1      |
+----------------+--------------------------------------------------------------------------------------------+-------------------+---------------+-------------+
| vl_paramak     | `docker://ibsim/vl_paramak <https://hub.docker.com/r/ibsim/vl_paramak>`_                   | |build-status_pa| | Paramak       | 0.8.6       |
+----------------+--------------------------------------------------------------------------------------------+-------------------+---------------+-------------+
| vl_openmc      | `docker://ibsim/vl_openmc <https://hub.docker.com/r/ibsim/vl_openmc>`_                     | |build-status_op| | OpenMC        | 0.13.2      |
+----------------+--------------------------------------------------------------------------------------------+-------------------+---------------+-------------+
| vl_paraview    | `docker://ibsim/vl_paraview <https://hub.docker.com/r/ibsim/vl_paraview>`_                 | |build-status_pv| | ParaView      | 5.11        |
+----------------+--------------------------------------------------------------------------------------------+-------------------+---------------+-------------+
| vl_modelib_v1  | `docker://ibsim/vl_modelib_v1 <https://hub.docker.com/r/ibsim/vl_modelib_v1>`_             | |build-status_mo| | MoDELib       | 1.0         |
+----------------+--------------------------------------------------------------------------------------------+-------------------+---------------+-------------+
| vl_iso2mesh    | `docker://ibsim/vl_iso2mesh <https://hub.docker.com/r/ibsim/vl_iso2mesh>`_                 | |build-status_is| | iso2mesh      | 1.9.6       |
+----------------+--------------------------------------------------------------------------------------------+-------------------+---------------+-------------+
| vl_cad2vox     | `docker://ibsim/vl_cad2vox <https://hub.docker.com/r/ibsim/vl_cad2vox>`_                   | |build-status_cv| | CAD2Vox       | 1.26        |
+----------------+--------------------------------------------------------------------------------------------+-------------------+---------------+-------------+
| vl_gvxr        | `docker://ibsim/vl_gvxr <https://hub.docker.com/r/ibsim/vl_gvxr>`_                         | |build-status_gv| | gVXR          | 2.0.2       |
+----------------+--------------------------------------------------------------------------------------------+-------------------+---------------+-------------+
| vl_cil         | `docker://ibsim/vl_cil <https://hub.docker.com/r/ibsim/vl_cil>`_                           | |build-status_ci| | CIL           | 22.1.0      |
+----------------+--------------------------------------------------------------------------------------------+-------------------+---------------+-------------+
| vl_salomemeca  | `docker://ibsim/vl_salomemeca <https://hub.docker.com/r/ibsim/vl_salomemeca>`_             | |build-status_sa| | Salome-Meca   | 2019.0.3    |
+----------------+--------------------------------------------------------------------------------------------+-------------------+---------------+-------------+
| vl_aster_v14_6 | `docker://ibsim/vl_aster_v14_6 <https://hub.docker.com/r/ibsim/vl_aster_v14_6>`_           | |build-status_as| | Code_Aster    | 14.6        |
+----------------+--------------------------------------------------------------------------------------------+-------------------+---------------+-------------+
| vl_coms_test   | `docker://ibsim/vl_coms_test <https://hub.docker.com/r/ibsim/vl_coms_test>`_               | |build-status_co| | Utils         | 1.0         |
+----------------+--------------------------------------------------------------------------------------------+-------------------+---------------+-------------+

.. |build-status_vl| image:: https://img.shields.io/docker/cloud/build/ibsim/virtuallab
.. |build-status_pa| image:: https://img.shields.io/docker/cloud/build/ibsim/vl_paramak
.. |build-status_op| image:: https://img.shields.io/docker/cloud/build/ibsim/vl_openmc
.. |build-status_pv| image:: https://img.shields.io/docker/cloud/build/ibsim/vl_paraview
.. |build-status_mo| image:: https://img.shields.io/docker/cloud/build/ibsim/vl_modelib_v1
.. |build-status_is| image:: https://img.shields.io/docker/cloud/build/ibsim/vl_iso2mesh
.. |build-status_cv| image:: https://img.shields.io/docker/cloud/build/ibsim/vl_cad2vox
.. |build-status_gv| image:: https://img.shields.io/docker/cloud/build/ibsim/vl_gvxr
.. |build-status_ci| image:: https://img.shields.io/docker/cloud/build/ibsim/vl_cil
.. |build-status_sa| image:: https://img.shields.io/docker/cloud/build/ibsim/vl_salomemeca
.. |build-status_as| image:: https://img.shields.io/docker/cloud/build/ibsim/vl_aster_v14_6
.. |build-status_co| image:: https://img.shields.io/docker/cloud/build/ibsim/vl_coms_test









References
**********
.. footbibliography::