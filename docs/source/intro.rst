Introduction
============

**VirtualLab** (VL) is a software package which enables the user to run Finite Element (FE) simulations of an increasing number of physical laboratory experiments, i.e. their 'virtual counterparts'.

The software is mostly written in python, and is fully parameterised such that it can be run in 'batch mode', i.e. non-interactively, via the command line. The bulk of the simulations are carried out in the FE solver `Code_Aster <https://www.code-aster.org/>`_.

The pre and post processing is carried out using the various software:

* `SALOME <https://www.salome-platform.org/>`_: Mesh generation
* `ERMES <https://ruben-otin.blogspot.com/2015/04/ruben-otin-software-ruben-otin-april-19.html>`_: Induction heating
* `ParaVis <https://docs.salome-platform.org/latest/dev/PARAVIS/>`_: Data visualisation

While this package has been written for use from the command line, some capabilities have been included to use the GUI for debugging and training.
