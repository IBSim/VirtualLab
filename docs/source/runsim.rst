Running a Simulation
====================

Launching VirtualLab
********************

Command Line Interface
######################

If **VirtualLab** has been installed correctly, the main program will have been added to your system :bash:`<path>`. In this case, it is possible to call **VirtualLab** from the terminal (also known as command line interface, CLI) or a bash script from any location in your system. To facilitate automation, **VirtualLab** has purposefully been designed to run without a graphical user interface (GUI).

.. _usage:

Usage of 'VirtualLab':
  VirtualLab -f <path>

Options:
   | :bash:`-f <path>` Where <path> points to the location of the python *Run* file (relative to the current working directory).
   | :bash:`-k <Name=Value>` Overwrite the value specified for variables/keyword arguments specified in the *Run* file.
   | :bash:`-h` Display the help menu.

* The default behaviour is to exit if no :bash:`<path>` is given.

The `RunFiles <structure.html#runfiles>`_ directory contains the *Run* files used to specify the settings of virtual experiments. These files contain the relevant information required to run analysis using **VirtualLab**, such as the type of virtual experiment or the mode in which it is run. 

Containers
##########

Launching **VirtualLab** with a container can be carried out in several ways:

* Interactively
* Non-interactively
* Batch mode

Containers are write-protected. This means that, although they have an internal file structure, the user must tell the container where to read input files and write outputs by mounting (or binding) directories. The examples shown here demonstrate how to achieve this.

Interactive
~~~~~~~~~~~

These commands are examples of how to shell in to the **VirtualLab** container for an interactive session. Once inside the container, then **VirtualLab** may be launched as normally from a CLI command, detailed in :ref:`'Usage' <usage>`.

**Docker** ::

  sudo docker run -it \
  --mount type=bind, \
  /tmp=/tmp, \
  /home/$USER/flasheur:/home/ibsim/flasheur, \
  /home/$USER/Input:/home/ibsim/VirtualLab/Intput, \
  /home/$USER/Output:/home/ibsim/VirtualLab/Output \
  ibsim/virtuallab:latest

**Singularity** ::

  singularity shell --contain --bind \
  /tmp, \
  /dev, \
  /home/$USER/flasheur:/home/ibsim/flasheur, \
  /home/$USER/Input:/home/ibsim/VirtualLab/Intput, \
  /home/$USER/Output:/home/ibsim/VirtualLab/Output \
  VirtualLab.sif

.. note::
  :bash:`$USER` should be replaced by the username of the host user.

  Code_Aster requires write access to the :file:`/tmp` and :file:`../flasheur` directories. The user should create a local :file:`../flasheur` directory before launching the container. 

  The :file:`..Input` directory is where the user's custom simulation files should be kept and :file:`..Output` is where the simulation results are written. These can be customised as desired on the host system.

Non-Interactive
~~~~~~~~~~~~~~~

To launch **VirtualLab** from outside the container, CLI commands as detailed in :ref:`'Usage' <usage>` must be sent as arguments.

**Docker** ::

  sudo docker run -it \
  --mount type=bind, \
  /tmp=/tmp, \
  /home/$USER/flasheur:/home/ibsim/flasheur, \
  /home/$USER/Input:/home/ibsim/VirtualLab/Intput, \
  /home/$USER/Output:/home/ibsim/VirtualLab/Output \
  ibsim/virtuallab:latest \
  "VirtualLab.sif -f /home/$USER/Input/RunFile.py"

**Singularity** ::

  singularity exec --contain --bind \
  /tmp, \
  /dev, \
  /home/$USER/flasheur:/home/ibsim/flasheur, \
  /home/$USER/Input:/home/ibsim/VirtualLab/Intput, \
  /home/$USER/Output:/home/ibsim/VirtualLab/Output \
  VirtualLab.sif -f /home/$USER/Input/RunFile.py

Batch Mode
~~~~~~~~~~

In batch mode, rather than launching the command directly it is normally entered within a script which is sent to a job scheduler (or workload manager). The command is then out in a queue to be executed when the requested resources become available. Singularity is often the platform of choice for shared HPC resources because it can be used without the user needing admin privileges. This is a Singularity example for the `slurm <https://slurm.schedmd.com/>`_ job scheduler on Supercomputing Wales's sunbird system. 

**Singularity** ::

  #!/bin/bash --login
  #SBATCH --job-name=VirtualLab
  #SBATCH --output=VL.out.%J
  #SBATCH --error=VL.err.%J
  #SBATCH --time=0-00:20
  #SBATCH --ntasks=16
  #SBATCH --mem-per-cpu=1000
  #SBATCH --ntasks-per-node=16

  module load singularity/3.6.3

  singularity exec --contain --bind \
  /tmp, \
  /dev, \
  /home/$USER/flasheur:/home/ibsim/flasheur, \
  /home/$USER/Input:/home/ibsim/VirtualLab/Intput, \
  /home/$USER/Output:/home/ibsim/VirtualLab/Output \
  VirtualLab.sif -f /home/$USER/Input/RunFile.py

Using a GUI
~~~~~~~~~~~

Although **VirtualLab** is predominantly set up to be used without a GUI, the user may sometimes wish to use the GUI for reasons such as checking meshes or simulation results interactively. In this event, it is possible to use either platform to launch the relevant software from within the container and interact with it as if it were installed on the local machine.

These commands are examples of how to launch the GUI interface of salome with the **VirtualLab** container.

**Docker** ::

  sudo docker run \
  --mount type=bind, \
  /tmp=/tmp, \
  /home/$USER/flasheur:/home/ibsim/flasheur, \
  /home/$USER/Input:/home/ibsim/VirtualLab/Intput, \
  /home/$USER/Output:/home/ibsim/VirtualLab/Output \
  --net=host --env="DISPLAY" \
  --volume="$HOME/.Xauthority:/root/.Xauthority:rw" \
  ibsim/virtuallab:latest salome

**Singularity** ::

  singularity exec --contain --bind \
  /tmp, \
  /dev, \
  /home/$USER/flasheur:/home/ibsim/flasheur, \
  /home/$USER/Input:/home/ibsim/VirtualLab/Intput, \
  /home/$USER/Output:/home/ibsim/VirtualLab/Output \
  salome

Virtual Machines
################

Once logged into the VM the user is presented with an Ubuntu desktop environment which can be used identically to a native Linux installation. That is, with the use of the CLI in a terminal **VirtualLab** may be launched as detailed in :ref:`'Usage' <usage>`.

Header
******

At the top of each *Run* file is the header, common for all analysis, which includes various commands e.g. importing libraries. ::

  #!/usr/bin/env python3
  
  import sys
  from os.path import dirname, abspath
  sys.dont_write_bytecode=True
  sys.path.append(dirname(dirname(abspath(__file__))))
  from Scripts.Common.VirtualLab import VLSetup

Setup
*****

Simulation
##########

Usage:
  Simulation = '$TYPE'

This is used to select the 'type' of virtual experiment to be conducted.

Types available:
   | ``Tensile``
   | ``LFA``
   | ``HIVE``

For further details on each simulation see `Virtual Experiments <virtual_exp.html#laser-flash-analysis>`_.

Project
#######
Usage:
  Project = '$USER_STRING'

User-defined field to specify the name of the project being worked on. 

All data for a project is stored in the project directory located at :file:`Output/$SIMULATION/$PROJECT`. Here you will find the sub-directory 'Meshes' which contain the meshes generated for the project, and a sub-directory for each *StudyName*, that is:

   | :file:`Output/$SIMULATION/$PROJECT/Meshes`
   | :file:`Output/$SIMULATION/$PROJECT/$StudyName`

StudyName
#########

Usage: 
  StudyName = '$USER_STRING'
  
User-defined field used to group together virtual experiments.

Simulation results will be stored in the *StudyName* sub-directory of the `Project`_.

For example, if a parameter sweep of a tensile load magnitude was being conducted as part of the additive-manufacturing project, *StudyName* could be called 'LoadMagnitudeSweep'.

Results from all simulations which are part of this 'study' will be saved in the directory :file:`Output/Tensile/additive-manufacturing/LoadMagnitudeSweep`

Parameters_Master
#################

Usage:
  Parameters_Master = '$FNAME'

Name of the file which includes values for all the required variables for the selected virtual experiment. This file must be in the directory :file:`Input/$SIMULATION/$PROJECT`.

.. note:: Do not include the '.py' file extension as part of $FNAME.

The variables in this file are assigned to two different ``Namespaces``; *Mesh* and *Sim*. A ``Namespace`` is essentially an empty class which attributes can be assigned to.  

Mesh
~~~~

Variables within this namespace define the parameters required by **SALOME** to construct a mesh, such as geometric dimensions or mesh fineness.

The script :file:`$MESH.FILE.py` is executed in **SALOME** using the attributes of *Mesh* to create the geometry and subsequent mesh. This script must be in directory :file:`Scripts/$SIMULATION/Mesh`.

The meshes will be stored in ``MED`` format under the name *Mesh.Name* in the 'Meshes' directory of the `Project`_, i.e. :file:`Output/$SIMULATION/$PROJECT/Meshes`.

Sim
~~~

Variables within this namespace define the parameters needed by **Code_Aster** to perform a FE simulation.

The script :file:`$SIM.ASTERFILE.py` is executed in **Code_Aster** using the attributes of *Sim* to initiate the simulation. This script must be in directory :file:`Scripts/$SIMULATION/Aster`

Optional pre and post-processing scripts can be run by specifying them in *Sim.PreAsterFile* and *Sim.PostAsterFile* respectively. These scripts, which are executed before and after the **Code_Aster** call, must be in directories :file:`Scripts/$SIMULATION/PreAster` and :file:`PostAster` respectively. 

Simulation information and data will be stored in the sub-directory *Sim.Name* of the directory *StudyName*

Parameters_Var
##############

Usage:
  Parameters_Var = {'$FNAME'/None}

Name of the file which includes value ranges for particular variables of the user's choice. These variables must be a sub-set of those in *Parameters_Master*. The values defined in this file will be used instead of those specified in *Parameters_Master*.

Value ranges for given variables are used to perform a parameterised 'study' where multiple simulations are conducted concurrently. 

This file must be in the same directory as the *ParametersMaster* file.

If *Parameters_Var* is set to :code:`None` a single simulation is run using the values defined in *Parameters_Master*. 

Please see the `Tutorials <examples.html>`_ to see this in action.

.. note:: Do not include the '.py' file extension as part of $FNAME.

Mode
####

Usage:
  mode = "$OPTION"

This dictates how much information is printed in the terminal during the running of **VirtualLab**. 

Options available:
   | ``Interactive`` Prints all output to the terminal.
   | ``Continuous`` Writes the output to a file as it is generated.
   | ``Headless`` Writes output to file at the end of the process.

.. note:: 'I'/'C'/'H' may be used in place of the full option names.

Environment
***********

.. class:: VLSetup

  The VLSetup class interfaces between the system, **SALOME** and **Code_Aster** to ensure that the full workflow of a virtual experiment can be completed solely via the command line. 

  .. attribute:: __init__(Simulation, Project,StudyName,Parameters_Master, Parameters_Var, Mode)

    The variables detailed above are passed as arguments, making it possible to differentiate between different virtual experiments and how results are to be stored.


  .. attribute:: Control(RunMesh=True, RunSim=True)

    This function is responsible for checking that all defined files exist in the expected location. These include *Parameters_Master* and *Parameters_Var* and the files specified therein  (``Mesh.File``, ``Sim.PreAsterFile``, ``Sim.AsterFile``, ``Sim.PostAsterFile``). Once this is satisfied, output directories are created for the results, and the necessary files are created to produce mesh(es) and run simulation(s).

      | ``RunMesh`` bool (optional)
      |   Indicates whether or not the meshing routine will be run. Default is True.
      | ``RunSim``  bool (optional)
      |   Indicates whether or not the simulation routine will be run. Default is True.

  .. attribute:: Mesh(NumThreads=1, ShowMesh=False, MeshCheck=None)

    This function is the meshing routine. The mesh(es) defined using ``Mesh`` in *Parameters_Master* and *Parameters_Var* are created and saved to the sub-directory 'Meshes' in the project directory along with a file detailing the variables used for their creation. If RunMesh is set to False in 'Control' then this routine is skipped. This may be useful when different simulation parameters are to be used on a pre-existing mesh

      | ``NumThreads`` int (optional)
      |   Number of meshes created simultaneously. The number specified will depend on the resources available, such as number of CPUs, RAM etc. Default is 1.
      | ``ShowMesh`` bool (optional)
      |   Indicates whether or not to open created mesh(es) in the **SALOME** GUI for visualisation to assess their suitability. VirtualLab will terminate once the GUI is closed and no simulation will be carried out. Default is False.
      | ``MeshCheck`` '$MESH_NAME' (optional)
      |   '$MESH_NAME' is constructed in the **SALOME** GUI for debugging. Default is None.


  .. attribute:: Sim(NumThreads=1, RunPreAster=True, RunAster=True, RunPostAster=True, ShowRes=False, memory=2, ncpus=1, mpi_nbcpu=1, mpi_nbnoeud=1)

    This function is the simulation routine. The simulation(s) defined using ``Sim`` in *Parameters_Master* and *Parameters_Var* are carried out with the results saved to the sub-directory '$STUDYNAME' in the project directory. This routine also runs the pre and post-processing scripts, if they are provided. If RunSim is set to False in 'Control' then this routine is skipped. 

      | ``NumThreads`` int (optional)
      |   Number of simulations run simultaneously. The number specified will depend on the resources available, such as number of CPUs, RAM etc. Default is 1.
      | ``RunPreAster`` bool (optional)
      |   Indicates whether or not to run the optional pre-processing script provided in `Sim.PreAsterFile`. Default is True.
      | ``RunAster`` bool (optional)
      |   Indicates whether or not to run the **Code_Aster** script provided in ``Sim.AsterFile``. Default is True.
      | ``RunPostAster`` bool (optional)
      |   Indicates whether or not to run the optional post-processing script provided in ``Sim.PostAsterFile``. Default is True.
      | ``ShowRes`` bool (optional)
      |   Visualises the .rmed results file(s) produced by **Code_Aster** through the **ParaVis** module in **SALOME**. Default is False.
      | ``memory`` float (optional)
      |   Number of GBs of memory allocated to **Code_Aster** for simulations. Default is 2.
      | ``ncpus`` int (optional)
      |   Number of processors used for the solver 'MULT_FRONT' in **Code_Aster**. This is the only solver with built in parallelism in the non-MPI version. Default is 1.
      | ``mpi_nbcpu`` int (optional)
      |   Number of cpus cores for MPI parallelism. Default is 1.
      | ``mpi_nbnoeud`` int (optional)
      |   Number of nodes which mpi_nbnoeud are spread over. Default is 1.

    .. note:: The binary distribution of standalone **Code_Aster** and the version packaged with **Salome-Meca** does not make use of MPI. To use MPI with **Code_Aster** it must be compiled from source, in which case the solvers 'MUMPS' and 'PETSC' may be used.

    .. note:: If **Code_Aster** has been compiled with MPI then ncpus specifies the number of OpenMP cpus used for each mpi cpu. For example if mpi_nbcpu=8 and ncpus=6, then 48 cpus are used.

  .. attribute:: Cleanup()

    This function removes all tmp directories created and closes any open instance of **SALOME**.



