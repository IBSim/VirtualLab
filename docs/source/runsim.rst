Running a Simulation
====================

The `RunFiles <structure.html#runfiles>`_ directory contains the driver files to launch virtual experiments. These files contain the relevant information required to run analysis using **VirtualLab**, such as the type of virtual experiment or the mode in which it is run.

Currently, no options can be passed to Run files from the command line. Options must be selected by editing the files directly. 

Header
******

At the top of each Run file is the header, common for all analysis, which includes various commands e.g. importing libraries. ::

  #!/usr/bin/env python3
  
  import sys
  from os.path import dirname, abspath
  sys.dont_write_bytecode=True
  sys.path.append(dirname(dirname(abspath(__file__))))
  from Scripts.Common.VirtualLab import VLSetup

Simulation
**********

Usage:
  Simulation = '$TYPE'

This is used to select the 'type' of virtual experiment to be conducted.

Types available:
   | ``Tensile``
   | ``LFA``
   | ``HIVE``

See below for further details.

Mechanical
##########

`Tensile <virtual_exp.html#tensile-testing>`_ is a standard mechanical tensile test where a 'dog-bone' shaped sample is loaded. The load can be applied as a constant force whilst measuring the displacement or as a constant displacement whilst measuring the required load. This provides information about mechanical properties such as Young's elastic modulus.

Thermal
#######

`Laser flash analysis <virtual_exp.html#laser-flash-analysis>`_ (LFA) is an experiment where a disc shaped sample has a short laser pulse incident on one surface, whilst the temperature change is tracked with respect to time on the opposing surface. This is used to measure thermal diffusivity, which is used to calculate thermal conductivity.

Multi-Physics
#############

`Heat by Induction to Verify Extremes <virtual_exp.html#hive>`_ (HIVE) is an experimental facility at the UK Atomic Energy Authority's (UKAEA) Culham site. It is used to expose plasma-facing components to the high thermal loads they will be subjected to in a fusion energy device. In this experiment, samples are thermally loaded on one surface by induction heating whilst being actively cooled with pressurised water.


Project
*******
Usage:
  Project = '$USER_STRING'

User-defined field to specify the name of the project being worked on. 

All data for a project is stored in the project directory located at :file:`Output/$SIMULATION/$PROJECT`. Here you will find the sub-directory 'Meshes' which contain the meshes generated for the project, and a sub-directory for each *StudyName*.


StudyName
*********

Usage: 
  StudyName = '$USER_STRING'
  
User-defined field used to group together virtual experiments.

Simulation results will be stored in the *StudyName* sub-directory of the `Project`_.

For example, if a parameter sweep of a tensile load magnitude was being conducted as part of the additive-manufacturing project, *StudyName* could be called 'LoadMagnitudeSweep'.

Results from all simulations which are part of this 'study' will be saved in directory :file:`Output/Tensile/additive-manufacturing/LoadMagnitudeSweep`

Parameters_Master
*****************

Usage:
  Parameters_Master = '$FNAME'

Name of the file which includes values for all the required variables for the selected virtual experiment. This file must be in the directory :file:`Input/$SIMULATION/$PROJECT`.

.. note:: Do not include the '.py' file extension as part of $FNAME.

The variables in this file are attributed to two different ``Namespaces``; *Mesh* and *Sim*. A ``Namespace`` is essentially an empty class which attributes can be assigned to.  

Mesh
####

Variables within this namespace define the parameters required by **SALOME** to construct a mesh, such as geometric dimensions or mesh fineness.

The script :file:`$MESH.FILE.py` is executed in **SALOME** using the attributes of *Mesh* to create the geometry and subsequent mesh. This script must be in directory :file:`Scripts/$SIMULATION/Mesh`.

The meshes will be stored in ``MED`` format under the name *Mesh.Name* in the 'Meshes' directory of the `Project`_.

Sim
###

Variables within this namespace define the parameters needed by **Code_Aster** to perform a FE simulation.

The script :file:`$SIM.ASTERFILE.py` is executed in **Code_Aster** using the attributes of *Sim* to initiate the simulation. This script must be in directory :file:`Scripts/$SIMULATION/Aster`

Optional pre and post-processing scripts can be run by specifying them in *Sim.PreAsterFile* and *Sim.PostAsterFile* respectively. These scripts, which are executed before and after the **Code_Aster** call, must be in directories :file:`Scripts/$SIMULATION/PreAster` and :file:`PostAster` respectively. 

Simulation information and data will be stored in the sub-directory *Sim.Name* of the directory *StudyName*

Parameters_Var
**************

Usage:
  Parameters_Var = {'$FNAME'/None}

Name of the file which includes value ranges for particular variables of the user's choice. These variables must be a sub-set of those in *Parameters_Master*. The values defined in this file will be used instead of those specified in *Parameters_Master*.

Value ranges for given variables are used to perform a parameterised 'study' where multiple simulations are conducted concurrently. 

This file must be in the same directory as the *ParametersMaster* file.

If *Parameters_Var* is set to :code:`None` a single simulation is run using the values defined in *Parameters_Master*. 

Please see the `Tutorials <examples.html>`_ to see this in action.

.. note:: Do not include the '.py' file extension as part of $FNAME.

Mode
****

Usage:
  mode = "$OPTION"

This dictates how much information is printed in the terminal during the running of **VirtualLab**. 

Options available:
   | ``Interactive`` Prints all output to the terminal.
   | ``Continuous`` Writes the output to a file as it is generated.
   | ``Headless`` Writes output to file at the end of the process.

.. note:: 'I'/'C'/'H' may be used in place of the full option names.

VLSetup
*******

.. class:: VLSetup

  The VLSetup class interfaces between the system, **SALOME** and **Code_Aster** to ensure that the full workflow of a virtual experiment can be completed solely via the command line. 

  .. attribute:: __init__(Simulation, Project,StudyName,Parameters_Master, Parameters_Var, Mode, port=None)

    The variables detailed above are passed as arguments, making it possible to differentiate between different virtual experiments and how results are to be stored.

      | ``port`` int (optional)
      |     Specify a port number on which **SALOME** is open. This will save the time required to open & close an instance of **SALOME** in **VirtualLab**. An instance is usually opened on ports starting at 2810. Default is None.

  .. attribute:: Create(RunMesh=True, RunSim=True)

    This function is responsible for checking that all defined files exist in the expected location. These include *Parameters_Master* and *Parameters_Var* and the files specified therein  (``Mesh.File``, ``Sim.PreAsterFile``, ``Sim.AsterFile``, ``Sim.PostAsterFile``). Once this is satisfied, output directories are created for the results, and the necessary files are created to produce mesh(es) and run simulation(s).

      | ``RunMesh`` bool (optional)
      |   Indicates whether or not the meshing routine will be run. Default is True.
      | ``RunSim``  bool (optional)
      |   Indicates whether or not the simulation routine will be run. Default is True.

  .. attribute:: Mesh(ShowMesh=False, MeshCheck=None)

    This function is the meshing routine. The mesh(es) defined using ``Mesh`` in *Parameters_Master* and *Parameters_Var* are created and saved to the sub-directory 'Meshes' in the project directory along with a file detailing the variables used for their creation. If RunMesh is set to False in 'Create' then this routine is skipped. This may be useful when different simulation parameters are to be used on a pre-existing mesh

      | ``ShowMesh`` bool (optional)
      |   Indicates whether or not to open created mesh(es) in the **SALOME** GUI for visualisation to assess their suitability. VirtualLab will terminate once the GUI is closed and no simulation will be carried out. Default is False.
      | ``MeshCheck`` '$MESH_NAME' (optional)
      |   '$MESH_NAME' is constructed in the **SALOME** GUI for debugging. Default is None.

  .. attribute:: Sim(RunPreAster=True,RunAster=True,RunPostAster=True,ShowRes=False,ncpus=1,memory=2,mpi_nbcpu=1,mpi_nbnoeud=1)

    This function is the simulation routine. The simulation(s) defined using ``Sim`` in *Parameters_Master* and *Parameters_Var* are carried out with the results saved to the sub-directory '$STUDYNAME' in the project directory. This routine also runs the pre and post-processing scripts, if they are provided. If RunSim is set to False in 'Create' then this routine is skipped. 

      | ``RunPreAster`` bool (optional)
      |   Indicates whether or not to run the optional pre-processing script provided in `Sim.PreAsterFile`. Default is True.
      | ``RunAster`` bool (optional)
      |   Indicates whether or not to run the **Code_Aster** script provided in ``Sim.AsterFile``. Default is True.
      | ``RunPostAster`` bool (optional)
      |   Indicates whether or not to run the optional post-processing script provided in ``Sim.PostAsterFile``. Default is True.
      | ``ShowRes`` bool (optional)
      |   Visualises the .rmed results file(s) produced by **Code_Aster** through the ParaVis module in **SALOME**. Default is False.
      | ``ncpus`` int (optional)
      |   Number of processors used by the solver 'MULT_FRONT' in **Code_Aster**. Default is 1.
      | ``memory`` float (optional)
      |   Number of GBs of memory allocated to **Code_Aster** for simulations. Default is 2.
      | ``mpi_nbcpu`` int (optional)
      |   Number of cpus cores for MPI parallelism. Default is 1.
      | ``mpi_nbnoeud`` int (optional)
      |   Number of nodes which mpi_nbnoeud are spread over. Default is 1.

    .. note:: The binary distribution of standalone **Code_Aster** and the version packaged with **Salome-Meca** does not make use of MPI. To use MPI with **Code_Aster** it must be compiled from source, in which case the solvers 'MUMPS' and 'PETSC' may be used.

    .. note:: ncpus and mpi_nbcpu will not conflict because only one value is used depending on the solver utilised. That is, if both variables are set, only one is passed to the solver.

  .. attribute:: Cleanup()

    This function removes all tmp directories created and closes the opened instance of **SALOME**.



