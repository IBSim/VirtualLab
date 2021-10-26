RunFile
====================

Header
******

At the top of each *RunFile* is the header, common for all analysis, which includes various commands e.g. importing libraries. ::

  #!/usr/bin/env python3

  import sys
  sys.dont_write_bytecode=True
  from Scripts.Common.VirtualLab import VLSetup

Setup
*****

The variables defined in the setup section are compulsory to launch **VirtualLab** successfully.

Simulation
##########

Usage:
  Simulation = '$TYPE'

This is used to select the 'type' of virtual experiment to be conducted.

Types available:
   | ``Tensile``
   | ``LFA``
   | ``HIVE``

For further details on each simulation see `Virtual Experiments <virtual_exp.html#virtual-experiments>`_.

Project
#######

Usage:
  Project = '$USER_STRING'

User-defined field to specify the name of the project being worked on.

All data for a project is stored in the project directory located at :file:`Output/$SIMULATION/$PROJECT`. Here you will find the sub-directory 'Meshes' which contain the meshes generated for the project, alongside results from simulations and data analysis conducted. The output generated would be:

   | :file:`Output/$SIMULATION/$PROJECT/Meshes/$Mesh.Name`
   | :file:`Output/$SIMULATION/$PROJECT/$Sim.Name`
   | :file:`Output/$SIMULATION/$PROJECT/$DA.Name`

Parameters_Master
#################

Usage:
  Parameters_Master = '$FNAME'

Name of the file which includes values for all the required variables for the selected virtual experiment. This file must be in the directory :file:`Input/$SIMULATION/$PROJECT`.

.. note:: Do not include the '.py' file extension as part of $FNAME.

The variables in this file are assigned to different ``Namespaces``; *Mesh*, *Sim* and *DA*. A ``Namespace`` is essentially an empty class that variables can be assigned to.

Mesh
~~~~

Variables within this namespace define the parameters required by **SALOME** to construct a mesh, such as geometric dimensions or mesh fineness.

The script :file:`$Mesh.File.py` is executed in **SALOME** using the attributes of *Mesh* to create the geometry and subsequent mesh. This script must be in directory :file:`Scripts/$SIMULATION/Mesh`.

The meshes will be stored in ``MED`` format under the name *Mesh.Name* in the 'Meshes' directory of the `Project`_, i.e. :file:`Output/$SIMULATION/$PROJECT/Meshes`.

Sim
~~~

Variables within this namespace define the parameters needed by **Code_Aster** to perform a FE simulation.

The command file :file:`$Sim.AsterFile.comm` is executed in **Code_Aster** using the attributes of *Sim* to initiate the simulation. This script must be in directory :file:`Scripts/$SIMULATION/Sim`.

Optional pre and post-processing scripts can be run by specifying them in *Sim.PreAsterFile* and *Sim.PostAsterFile* respectively. These scripts, which are executed before and after the **Code_Aster** are also found in :file:`Scripts/$SIMULATION/Sim`.

Simulation information and data will be stored in the sub-directory *Sim.Name* of the project directory, i.e. :file:`Output/$SIMULATION/$PROJECT/$Sim.Name`.

DA
~~~

Variables within this namespace define the parameters needed to perform data analysis (DA) on the data collected from simulations. These are generally python scipts. These files can be found in :file:`Scripts/$SIMULATION/DA`.

Like with the simulations, results for the data analysis is saved to :file:`Output/$SIMULATION/$PROJECT/$DA.Name`.

.. note:: *Sim.Name* and *DA.Name* can be written as paths to save in to sub folders of a project directory, i.e. *Sim.Name* = 'Test/Simulation' will create a sub-directory 'Test' in the project directory.


Parameters_Var
##############

Usage:
  Parameters_Var = {'$FNAME'/None}

Name of the file which includes value ranges for particular variables of the user's choice. These variables are usually a sub-set of those in *Parameters_Master*, with the values defined in this file used instead of those specified in *Parameters_Master*. Value ranges for given variables are used to perform parametric analyses, where multiple 'studies' are conducted.

As in *Parameters_Master*, values will be assigned to the ``Namespaces`` *Mesh*, *Sim* and *DA*. This file is also in :file:`Input/$SIMULATION/$PROJECT`.

If *Parameters_Var* is set to :code:`None` a single simulation is run using the values defined in *Parameters_Master*.

Please see the `Tutorials <examples.html>`_ to see this in action.

.. note:: Do not include the '.py' file extension as part of $FNAME.


Environment
***********

The environment section is where **VirtualLab** runs and produces results. There are optional keyword arguments (often referred to as kwargs) at different stages that will alter the way in which **VirtualLab** is run.

.. class:: VLSetup

  The VLSetup class interfaces between the system, **Python** and any integrated software packages, which are currently **SALOME**, **Code_Aster** and **ERMES**. This ensures that the full workflow of a virtual experiment can be completed solely via the command line.

  .. attribute:: __init__(Simulation, Project)

    This function initiates the VirtualLab class and defines key paths and variables required.

      | ``Simulation`` '$TYPE' (str)
      |   See `Simulation <runsim.html#simulation>`_
      | ``Project`` '$USER_STRING' (str)
      |   See `Project <runsim.html#project>`_

  .. attribute:: Settings(Mode='Headless', Launcher='Process', NbThreads=1)

    This is an optional step where VirtualLab settings can be changed.

      | ``Mode`` '$TYPE' (str, optional)
      |   This dictates how much information is printed in the terminal during the running of **VirtualLab**. Options available are:
      |   'Interactive' - Prints all output to individual pop-up terminals.
      |   'Terminal' - Prints all information to a single terminal.
      |   'Continuous'  - Writes the output to a file as it is generated.
      |   'Headless'  - Writes output to file at the end of the process. (Default)
      | ``Launcher`` '$TYPE' (str, optional)
      |   This defines the method used to launch the VirtualLab study. Currently available options are:
      |   'Sequential' - Each operation is run sequentially (no parallelism).
      |   'Process' - Parallelism for a single node only. (Default)
      |   'MPI' - Parallelism over multiple nodes.
      | ``NbThreads`` '$INTEGER' (int, optional)
      |   Defines how many of the studies that will run concurrently when using either the 'process' or 'MPI' launcher. Default is 1.

  .. attribute:: Parameters(Parameters_Master, Parameters_Var, RunMesh=True, RunSim=True, RunDA=True)

     This function creates the parameter files used by VirtualLab and defines information used by Mehs, Sim and DA. It is also responsible for checking that all defined files exist in the expected location, such as Parameters_Master, Parameters_Var and the files specified therein (Mesh.File, Sim.AsterFile etc.).

      | ``Parameters_Master`` ‘$FNAME’ (str)
      |   See `Parameters Master <runsim.html#parameters-master>`_
      | ``Parameters_Var`` {‘$FNAME’/None} (str)
      |   See `Parameters Var <runsim.html#parameters-var>`_
      | ``RunMesh`` bool (optional)
      |   Indicates whether or not the meshing routine will be run. Default is True.
      | ``RunSim``  bool (optional)
      |   Indicates whether or not the simulation routine will be run. Default is True.
      | ``RunDA``  bool (optional)
      |   Indicates whether or not the data analysis routine will be run. Default is True.

  .. attribute:: Mesh(ShowMesh=False, MeshCheck=None)

    This function is the meshing routine. The mesh(es) defined using ``Mesh`` in *Parameters_Master* and *Parameters_Var* are created and saved to the sub-directory 'Meshes' in the project directory along with a file detailing the variables used for their creation. If RunMesh is set to False in 'Parameters' then this routine is skipped. This may be useful when different simulation parameters are to be used on a pre-existing mesh

      | ``ShowMesh`` bool (optional)
      |   Indicates whether or not to open created mesh(es) in the **SALOME** GUI for visualisation to assess their suitability. VirtualLab will terminate once the GUI is closed and no simulation will be carried out. Default is False.
      | ``MeshCheck`` '$MESH_NAME' (optional)
      |   '$MESH_NAME' is constructed in the **SALOME** GUI for debugging. Default is None.


  .. attribute:: Sim(RunPreAster=True, RunAster=True, RunPostAster=True, ShowRes=False)

    This function is the simulation routine. The simulation(s) defined using ``Sim`` in *Parameters_Master* and *Parameters_Var* are carried out with the results saved to the project directory. This routine also runs the pre and post-processing scripts, if they are provided. If RunSim is set to False in 'Parameters' then this routine is skipped.

      | ``RunPreAster`` bool (optional)
      |   Indicates whether or not to run the optional pre-processing script provided in `Sim.PreAsterFile`. Default is True.
      | ``RunAster`` bool (optional)
      |   Indicates whether or not to run the **Code_Aster** script provided in ``Sim.AsterFile``. Default is True.
      | ``RunPostAster`` bool (optional)
      |   Indicates whether or not to run the optional post-processing script provided in ``Sim.PostAsterFile``. Default is True.
      | ``ShowRes`` bool (optional)
      |   Visualises the .rmed results file(s) produced by **Code_Aster** through the **ParaVis** module in **SALOME**. Default is False.

 .. attribute:: DA()

     This function is the data analysis routine. The analysis, defined using the namespace 'DA' in Parameters_Master and Parameters_Var, are carried out. The results are saved to Output/$SIMULATION/$PROJECT. If
     RunDA is set to False in VirtualLab.Parameters then this routine is skipped.

  .. attribute:: Cleanup()

    This function removes all tmp directories created and closes any open instance of **SALOME**.
