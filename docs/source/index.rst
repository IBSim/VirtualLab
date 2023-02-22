.. VirtualLab documentation master file, created by
   sphinx-quickstart on Thu Jun  4 21:58:03 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. role:: bash(code)
  :language: bash

:hide-toc:

Welcome to VirtualLab's documentation!
======================================

.. image:: https://gitlab.com/ibsim/media/-/raw/master/images/VirtualLab/VirtualLab_Logo.png
  :width: 800
  :alt: VirtualLab Logo
  :align: center

**VirtualLab** is a modular platform which enables the user to run simulations of physical laboratory experiments, i.e., their 'virtual counterparts'.

The motivation for creating a virtual laboratory is manyfold, for example:

* Planning and optimisation of physical experiments.
* Ability to directly compare experimental and simulation data, useful to better understand both physical and virtual methods.
* Augment sparse experimental data with simulation data for increased insight.
* Generating synthetic data to train machine learning models.

The software is mostly written in python, and is fully parametrised such that it can be run in 'batch mode', i.e., non-interactively, via the command line. This is in order to facilitate automation and so that many virtual experiments can be conducted in parallel.

Due to the modularity of the platform, by nature, **VirtualLab** is continually expanding. The bulk of the 'virtual experiments' currently included are carried out in the FE solver `Code_Aster <https://www.code-aster.org/>`_. However, there are also modules to simulate `X-ray computed tomography <https://gvirtualxray.fpvidal.net/>`_, `irradiation damage of materials <https://github.com/giacomo-po/MoDELib>`_ and `electromagnetics <https://ruben-otin.blogspot.com/2015/04/ruben-otin-software-ruben-otin-april-19.html>`_.

The pre and post processing is carried out using various software, for example:

* `SALOME <https://www.salome-platform.org/>`_: Mesh generation
* `Cad2Vox <https://github.com/bjthorpe/Cad2vox>`_: Mesh voxelisation
* `CIL <https://ccpi.ac.uk/cil/>`_: CT reconstruction
* `SuRVoS <https://github.com/DiamondLightSource/SuRVoS2>`_: Image segmentation
* `iso2mesh <http://iso2mesh.sourceforge.net/>`_: Image-based meshing
* `PyTorch <https://pytorch.org/>`_: Data analytics
* `ParaVis <https://docs.salome-platform.org/latest/dev/PARAVIS/>`_: Data visualisation
  
While this platform has been written for use from the command line, some capabilities have been included to use GUIs offered by the various software for debugging and training.

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
| vl_monolith    | `docker://ibsim/virtuallab_monolith <https://hub.docker.com/r/ibsim/virtuallab_monolith>`_ | |build-status_mn| | Various       | N/A         |
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
.. |build-status_mn| image:: https://img.shields.io/docker/cloud/build/ibsim/virtuallab_monolith


.. toctree::
   :caption: Contents:
   :hidden:
   :maxdepth: 1

   Introduction <index>


.. toctree::
   :hidden:
   :maxdepth: 2

   install
   structure
   virtual_exp
   runsim/index
   examples/index
   contributing
   about