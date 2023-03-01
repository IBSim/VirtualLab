The RunFile Explained
=====================

The *RunFile* contains all the necessary information to launch analyses using **VirtualLab**. The *RunFile* is executed in strict order such that it is possible to build up a complex workflow with conditional dependencies. This does mean that it is important to carefully consider the order of the *RunFile* sections and sub-sections.

Template
********

.. admonition:: Template
   :class: action

   A template *RunFile* for **VirtualLab**::

        #!/usr/bin/env python3
        #===============================================================================
        # Header
        #===============================================================================

        import sys
        sys.dont_write_bytecode=True
        from Scripts.Common.VirtualLab import VLSetup
        
        #===============================================================================
        # Definitions
        #===============================================================================
        
        Simulation='$TYPE'
        Project='$USER_STRING'
        Parameters_Master='$FNAME'
        Parameters_Var='$FNAME'/None
        
        #===============================================================================
        # Environment
        #===============================================================================

        VirtualLab=VLSetup(
                   Simulation,
                   Project
                   )

        VirtualLab.Settings(
                   Mode='$TYPE',
                   Launcher='$TYPE',
                   NbJobs=$INTEGER
                   )

        VirtualLab.Parameters(
                   Parameters_Master,
                   Parameters_Var,
                   RunMesh=bool,
                   RunSim=bool,
                   RunDA=bool,
                   RunVoxelise=bool
                   )
        
        #===============================================================================
        # Methods
        #===============================================================================

        VirtualLab.Mesh(
                   ShowMesh=bool,
                   MeshCheck='$MESH_NAME'/None
                   )

        VirtualLab.Sim(
                   RunPreAster=bool,
                   RunAster=bool,
                   RunPostAster=bool,
                   ShowRes=bool
                   )

        VirtualLab.DA()

        VirtualLab.Voxelize()
        

Header
******

At the top of each *RunFile* is the header, common for all analyses, which includes various commands e.g. importing libraries. It is unlikely that you will need to amend this section. ::

  #!/usr/bin/env python3
  #===============================================================================
  # Header
  #===============================================================================
  
  import sys
  sys.dont_write_bytecode=True
  from Scripts.Common.VirtualLab import VLSetup

Definitions
***********

Following this is the definitions section, where variables are defined which are compulsory to launch **VirtualLab** successfully.

Simulation
~~~~~~~~~~

.. _usage:

Usage:
::
  
  Simulation = '$TYPE'

This is used to select the 'type' of virtual experiment to be conducted.

Types available:
   | ``Tensile``
   | ``LFA``
   | ``HIVE``

For further details on each simulation see `Virtual Experiments <../virtual_exp.html#virtual-experiments>`_.

Project
~~~~~~~

Usage:
::
  
  Project = '$USER_STRING'

User-defined field to specify the name of the project being worked on.

All data for a project is stored in the project directory located at :file:`Output/$SIMULATION/$PROJECT`. Here you will find the sub-directory 'Meshes' which contain the meshes generated for the project, alongside results from simulations and data analyses conducted. The output generated would be:

   | :file:`Output/$SIMULATION/$PROJECT/Meshes/$Mesh.Name`
   | :file:`Output/$SIMULATION/$PROJECT/$Sim.Name`
   | :file:`Output/$SIMULATION/$PROJECT/$DA.Name`

Parameters_Master
~~~~~~~~~~~~~~~~~

Usage:
::
  
  Parameters_Master = '$FNAME'

Name of the file which includes values for all the required variables for the selected virtual experiment. This file must be in the directory :file:`Input/$SIMULATION/$PROJECT`.

.. note:: Do not include the '.py' file extension as part of $FNAME.

The variables in this file are assigned to different ``Namespaces``, which is essentially an empty class that variables can be assigned to.

Mesh
####
The ``Mesh`` namespace defines the parameters required by **SALOME** to construct a mesh, such as geometric dimensions or mesh fineness. The script :file:`$Mesh.File.py` is executed in **SALOME** using the attributes of ``Mesh`` to create the geometry and subsequent mesh. This script must be in directory :file:`Scripts/Experiments/$SIMULATION/Mesh`. The meshes will be stored in ``MED`` format under the name ``Mesh.Name`` in the 'Meshes' directory of the `Project`_, i.e. :file:`Output/$SIMULATION/$PROJECT/Meshes`.

Sim
###
The ``Sim`` namespace define the parameters needed by **Code_Aster** to perform a FE simulation. The command file :file:`$Sim.AsterFile.comm` is executed in **Code_Aster** using the attributes of ``Sim`` to initiate the simulation. This script must be in directory :file:`Scripts/Experiments/$SIMULATION/Sim`. Optional pre- and post-processing scripts can be run by specifying them in ``Sim.PreAsterFile`` and ``Sim.PostAsterFile`` respectively. These scripts, which are executed before and after the **Code_Aster** are also found in :file:`Scripts/Experiments/$SIMULATION/Sim`. Simulation information and data will be stored in the sub-directory ``Sim.Name`` of the project directory, i.e. :file:`Output/$SIMULATION/$PROJECT/$Sim.Name`.

DA
###
The ``DA`` namespace define the parameters needed to perform data analyses (DA) on the data collected from simulations. These are generally python scripts. These files can be found in :file:`Scripts/Experiments/$SIMULATION/DA`. Like with the simulations, results for the data analyses are saved to :file:`Output/$SIMULATION/$PROJECT/$DA.Name`.

.. note:: ``Mesh.Name``, ``Sim.Name`` and ``DA.Name`` can be written as paths to save in to sub folders of a project directory, i.e. ``Sim.Name`` = 'Test/Simulation' will create a sub-directory 'Test' in the project directory.


Parameters_Var
~~~~~~~~~~~~~~

Usage:
::
  
  Parameters_Var = {'$FNAME'/None}

Name of the file which includes value ranges for particular variables of the user's choice. This is used in tandem with `Parameters_Master`_.

Variables defined here are usually a sub-set of those in *Parameters_Master*, with the values specified here overwriting those in the master.

Value ranges for given variables are used to perform parametric analyses, where multiple 'studies' are conducted.

As in *Parameters_Master*, values will be assigned to the ``Namespaces`` ``Mesh``, ``Sim`` and ``DA``. This file is also in :file:`Input/$SIMULATION/$PROJECT`.

If set to :code:`None` a single study is run using the values defined in *Parameters_Master*.

Please see the `Tutorials <../examples/index.html>`_ to see this in action.

.. note:: Do not include the '.py' file extension as part of $FNAME.

Environment
***********

The next section is for setting the **VirtualLab** environment. That is, how the user would like to interact with **VirtualLab** and how it should make use of the available hardware. It is necessary to create the environment before starting any `Methods`_. However, it is possible to change the envrionment later in the *RunFile* as part of the workflow. For example, it may be desirable to only have a single job during meshing but multiple jobs for the simulation if performing a parameter sweep of boundary conditions with the same geometry.

VLSetup
~~~~~~~

``VLSetup`` takes the previously set `Definitions`_ to start building the environment. It is unlikely that you will need to amend this section. ::

    VirtualLab=VLSetup(
               Simulation,
               Project
               )

``VirtualLab.Settings``
~~~~~~~~~~~~~~~~~~~~~~~
This is an optional attribute of **VirtualLab** where settings can be changed. ::

    VirtualLab.Settings(
               Mode='Headless',
               Launcher='Process',
               NbJobs=1
               )

Mode
####

Usage:
::
  
  Mode = '$TYPE' (str, optional)

This dictates how much information is printed in the terminal during the running of **VirtualLab**. Options available are:

*   'Interactive' - Prints all output to individual pop-up terminals (currently not in use due to a change in X-Window implementation).
*   'Terminal' - Prints all information to a single terminal.
*   'Continuous'  - Writes the output to a file as it is generated.
*   'Headless'  - Writes output to file at the end of the process. (Default)

Launcher
########

Usage:
::
  
  Launcher = '$TYPE' (str, optional)

This defines the method used to launch the **VirtualLab** study. Currently available options are:

*   'Sequential' - Each operation is run sequentially (no parallelism).
*   'Process' - Parallelism for a single node only. (Default)
*   'MPI' - Parallelism over multiple nodes.


NbJobs
######

Usage:
::
  
  NbJobs = $INTEGER (int, optional)

Defines how many of the studies that will run concurrently when using either the 'process' or 'MPI' launcher. Default is 1.


``VirtualLab.Parameters``
~~~~~~~~~~~~~~~~~~~~~~~~~

This function creates the parameter files defined using `Parameters_Master`_ and `Parameters_Var`_. It also performs some checks, such as checking defined files exist in their expected locations, i.e., *Parameters_Master*, *Parameters_Var* and the files specified therein (Mesh.File, Sim.AsterFile etc.). ::

    VirtualLab.Parameters(
               Parameters_Master,
               Parameters_Var,
               RunMesh=True,
               RunSim=True,
               RunDA=True,
               RunVoxelise=True
               )


In addition to the parameter files and performing checks of associated file, it is possible to define whether particular `Methods`_ should run or not. By default, any method which is included in the later method section will run unless explicitly defined not to here.

Usage:
::
  
  Run$METHOD = bool (optional)

Indicates whether or not the method will be run. Default is :code:`True`. Currently available options are:

*   Mesh - For geometry creation and meshing.
*   Sim - For running simulations.
*   DA  - For data analysis of results.
*   Vox  - For voxelisation of meshes.

Methods
*******

This section is where the bulk of the activity of **VirtualLab** occurs. That is, until now, we have only put in place the necessary information to initiate a task. The methods section controls precisely which tasks **VirtualLab** will perform. They can be simple one step sequential tasks or highly complex parallelised tasks making use of multiple software packages.

``VirtualLab.Mesh``
~~~~~~~~~~~~~~~~~~~

This is the meshing routine. In fact, this routine first generates the CAD geometry from a set of parameters and then meshes it ready for simulation. The mesh(es) defined using ``Mesh`` in *Parameters_Master* and *Parameters_Var* are created and saved to the sub-directory 'Meshes' in the project directory along with a file detailing the variables used for their creation. If RunMesh is set to :code:`False` in `VirtualLab.Parameters`_ then this routine is skipped. This may be useful when different simulation parameters are to be used on a pre-existing mesh. ::

    VirtualLab.Mesh(
               ShowMesh=False,
               MeshCheck=None
               )


ShowMesh
########

Usage:
::
  
  ShowMesh = bool (optional)

Indicates whether or not to open created mesh(es) in the **SALOME** GUI for visualisation to assess their suitability. **VirtualLab** will terminate once the GUI is closed and no simulation will be carried out. Default is :code:`False`.

MeshCheck
#########

Usage:
::
  
  MeshCheck = '$MESH_NAME'/None (optional)

'$MESH_NAME' is constructed in the **SALOME** GUI for debugging. Default is None.

``VirtualLab.Sim``
~~~~~~~~~~~~~~~~~~

This function is the simulation routine. The simulation(s) defined using ``Sim`` in *Parameters_Master* and *Parameters_Var* are carried out with the results saved to the project directory. This routine also runs the pre- and post-processing scripts, if they are provided. If RunSim is set to :code:`False` in `VirtualLab.Parameters`_ then this routine is skipped. ::


    VirtualLab.Sim(
               RunPreAster=True,
               RunAster=True,
               RunPostAster=True,
               ShowRes=False
               )


RunPreAster
###########

Usage:
::
  
  RunPreAster = bool (optional)

Indicates whether or not to run the optional pre-processing script provided in `Sim.PreAsterFile`. Default is :code:`True`.

RunAster
########

Usage:
::
  
  RunAster = bool (optional)

Indicates whether or not to run the **Code_Aster** script provided in ``Sim.AsterFile``. Default is :code:`True`.

RunPostAster
############

Usage:
::
  
  RunPostAster = bool (optional)

Indicates whether or not to run the optional post-processing script provided in ``Sim.PostAsterFile``. Default is :code:`True`.

ShowRes
#######

Usage:
::
  
  ShowRes = bool (optional)

Visualises the .rmed results file(s) produced by **Code_Aster** through the **ParaVis** module in **SALOME**. Default is :code:`False`.

``VirtualLab.DA``
~~~~~~~~~~~~~~~~~

This function is the data analysis routine. The analyses, defined using the namespace ``DA`` in *Parameters_Master* and *Parameters_Var*, are carried out. The results are saved to :file:`Output/$SIMULATION/$PROJECT`. If RunDA is set to :code:`False` in `VirtualLab.Parameters`_ then this routine is skipped.

``VirtualLab.Voxelize``
~~~~~~~~~~~~~~~~~~~~~~~

This function is the routine to call **Cad2Vox**. The parameters used for the Voxelization process are defined in the namespace ``Vox`` in *Parameters_Master* and *Parameters_Var*. The resultant output images are saved to :file:`Output/$SIMULATION/$PROJECT/Voxel-Images`. If RunVoxelise is set to :code:`False` in `VirtualLab.Parameters`_ then this routine is skipped.

Example
*******

.. admonition:: Example
   :class: action

   An example *RunFile* for **VirtualLab** which will run a virtual tensile test::

        #!/usr/bin/env python3
        #===============================================================================
        # Header
        #===============================================================================

        import sys
        sys.dont_write_bytecode=True
        from Scripts.Common.VirtualLab import VLSetup
        
        #===============================================================================
        # Definitions
        #===============================================================================
        
        Simulation='Tensile'
        Project='Tutorials'
        Parameters_Master='TrainingParameters'
        Parameters_Var=None
        
        #===============================================================================
        # Environment
        #===============================================================================

        VirtualLab=VLSetup(
                   Simulation,
                   Project
                   )

        VirtualLab.Settings(
                   Mode='Terminal',
                   Launcher='Process',
                   NbJobs=1
                   )

        VirtualLab.Parameters(
                   Parameters_Master,
                   Parameters_Var,
                   RunMesh=True,
                   RunSim=True,
                   RunDA=False,
                   RunVoxelise=False
                   )
        
        #===============================================================================
        # Methods
        #===============================================================================

        VirtualLab.Mesh()

        VirtualLab.Sim(
                   ShowRes=True
                   )

        VirtualLab.DA()

        VirtualLab.Voxelize()
        