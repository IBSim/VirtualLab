Code Structure
==============

**VirtualLab** contains a number of essential directories needed for running VL simulations:

 * `Scripts`_
 * `Materials`_
 * `Input`_

In addition to this, other directories are included within the **VirtualLab** structure.

 * `docs`_
 * `RunFiles`_
 * `Output`_

Scripts
*******

This directory includes the scripts needed to initialise a **VirtualLab** simulation. 

Sub-directories within Scripts:

 * `Common`_
 * `Install`_
 * `VL Experiments`_

Common
######

*'Common'* contains the scripts needed by all simulation types. The perform tasks to set up the environment such as creating directories and inerfacing with **SALOME** and **Code_Aster**.

Install
#######

*'Install'* contains the scripts used by the `non-interactive installation <install.html#non-interactive-installation>`_, which will install and configure **VirtualLab** and its dependencies such as python, **Code_Aster**, **SALOME** and **ERMES**.

VL Experiments
##############

There are also a sub-directories for each simulation type for various physical laboratory experiments (e.g. Tensile, LFA and HIVE).

Inside each of these sub-directories are the relevant files required to run that specific virtual experiment. In the Mesh sub-directory you will find **SALOME** python scripts which create the mesh of the testpiece,  while the Aster sub-directory contains **Code_Aster** command scripts which outline the steps followed to setup the FE simulation, such as assigning materials properties to the testpiece.

These directories may also contain PreAster and PostAster sub-directories containing scripts which provide pre and post-processing capabilities, along with other simulation-specific sub-directories, e.g. ‘Laser’ for LFA which contains different laser pulse profiles measured experimentally.

Materials
*********

*'Materials'* contains the material properties used for FE simulations. Each sub-directory contains properties for different materials.

Material properties can be set to be linear or non-linear (e.g. temperature dependence).

The structure of the contents of this directory will be updated soon.

Input
*****

*'Input'* contains the parameters which will be used for running simulations, such as dimensions to create meshes and boundary conditions and materials for FE simulations.

Input has a sub-directory for each virtual experiment (further details in `Virtual Experiments <virtual_exp.html>`_), and within each of those you will find sub-directories for each different 'Project'. ::

  Input/$SIMULATION/$PROJECT

The 'Project' sub-directories contain the *'Parameters_Master'* and *'Parameters_Var'* files, explained further in `Running a Simulation <runsim.html>`_.

docs
****

The files required to create this documentation.

RunFiles
********

*'RunFiles'* is the directory that contain the driver files to launch virtual experiments. This directory contains a number of template files which the user may customise for their own applications.

A detailed template file *‘Run.py’* is included in the top level directory of **VirtualLab** i.e. the installation location.

The structure of the RunFile is explained in `Running a Simulation <runsim.html>`_.

Output
******

This directory will be made when the first **VirtualLab** scripts are run that create output files and will hold all data generated. This will include things such as: meshes; simulation results; visualisation images; analysis reports.

The *'Output'* directory has a similar sub-directory structure to *'Input'*. That is, it contains a sub-directory for each virtual experiment, and within each of those you will find sub-directories for different 'Projects'. ::

  Input/$SIMULATION/$PROJECT

Projects can contain more than one set of simulation results, called 'studies'. The specific output for each of these is stored in its own sub-directory. ::

  Input/$SIMULATION/$PROJECT/$STUDYNAME

All meshes used for a specific Project can be found in the 'Meshes' sub-directory. ::

  Input/$SIMULATION/$PROJECT/Meshes

