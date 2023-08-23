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

.. image:: https://gitlab.com/ibsim/media/-/raw/master/images/VirtualLab/Intro_01.png
  :width: 600
  :alt: VirtualLab Logo
  :align: center

The pre and post processing is carried out using various software, for example:

* `SALOME <https://www.salome-platform.org/>`_: Mesh generation
* `Cad2Vox <https://github.com/bjthorpe/Cad2vox>`_: Mesh voxelisation
* `CIL <https://ccpi.ac.uk/cil/>`_: CT reconstruction
* `SuRVoS <https://github.com/DiamondLightSource/SuRVoS2>`_: Image segmentation
* `iso2mesh <http://iso2mesh.sourceforge.net/>`_: Image-based meshing
* `PyTorch <https://pytorch.org/>`_: Data analytics
* `ParaVis <https://docs.salome-platform.org/latest/dev/PARAVIS/>`_: Data visualisation
  
While this platform has been written for use from the command line, some capabilities have been included to use GUIs offered by the various software for debugging and training.


.. toctree::
   :caption: Contents:
   :hidden:

   install
   containers
   structure
   virtual_exp
   runsim/index
   examples/index
   contributing
   about