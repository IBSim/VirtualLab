Code Structure
==============

**VirtualLab** contains a number of essential directories needed for running simulations:

 * `Scripts`_
 * `Materials`_
 * `Input`_

In addition to this, other directories are included within the **VirtualLab** structure.

 * `docs`_
 * `RunFiles`_
 * `Output`_

Scripts
*******

This directory includes the scripts needed to install **VirtualLab** and initialise a simulation. The content of the sub-directories are detailed below.

Install
#######

This contains the scripts used by the `non-interactive installation <install.html#non-interactive-installation>`_, which will install and configure **VirtualLab** and its dependencies such as python, **Code_Aster**, **SALOME** and **ERMES**.

Common
######

This contains the scripts needed by all simulation types. These includes setting up the environment through creating directories and inerfacing with **SALOME** and **Code_Aster**.

Simulations
###########

There are also a sub-directories for each simulation type for various physical laboratory experiments. Currently you will find `Tensile <virtual_exp.html#tensile-testing>`_, `LFA <virtual_exp.html#laser-flash-analysis>`_ and `HIVE <virtual_exp.html#hive>`_. 

Inside each of these are sub-directories containing the relevant files required to run that specific virtual experiment. In *Mesh* you will find **SALOME** python scripts which create the mesh of the testpiece, while the **Code_Aster** command scripts which outline the steps followed to setup the FE simulation can be found in *Aster*.

The sub-directories *PreAster* and *PostAster* contain scripts which provide pre and post-processing capabilities, if there are any. Simulation-specific sub-directories may also be included here, such as *Laser* for the LFA simulation which contains different laser pulse profiles measured experimentally.

Materials
*********

This directory contains the material properties used for FE simulations. Each sub-directory contains properties for different materials.

Material properties can be set to be linear or non-linear (e.g. temperature dependence).

The structure of the contents of this directory will be updated soon.

Input
*****

*Input* contains the parameters which will be used for running simulations, such as dimensions to create meshes and boundary conditions and materials for FE simulations.

Input has a sub-directory for each simulation type, and within each of those you will find sub-directories for each different 'Project'.

Here you will find :file:`$PARAMETERS_MASTER.py` and :file:`$PARAMETERS_VAR.py` files, which are explained in more detail in `Running a Simulation <runsim.html>`_.

docs
****

The files required to create this documentation.

RunFiles
********

The *RunFiles* directory contain the driver files to launch virtual experiments, referred to as *Run* files. This directory contains a number of templates which the user may customise for their own applications.

The structure of a *Run* file is explained in `Running a Simulation <runsim.html>`_. A detailed template file :file:`Run.py` is also included in the top level directory of **VirtualLab** i.e. the installation location.

Output
******

This directory will be created when the first **VirtualLab** scripts are run which produce output files. 

Similarly to the structure of `Input`_, this directory will have a sub-directory for each 'Project' within each simulation type. This directory will hold all data generated for the 'Project', such as: meshes; simulation results; visualisation images; analysis reports. The structure of the project directory is detailed in `Running a Simulation <runsim.html>`_.


