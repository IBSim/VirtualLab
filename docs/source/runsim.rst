Running a Simulation
====================

The `RunFiles <structure.html#runfiles>`_ directory contains the driver files to launch virtual experiments. These files contain the relevant information required to run analysis using **VirtualLab**, such as the type of virtual experiment or the mode in which it is run.

Currently, no options can be passed to 'Run Files' from the command line. Options must be selected by editing the files directly.

Header
******

At the top of each 'Run File' is the header, common for all 'studies', which includes various commands e.g. importing libraries. ::

  #!/usr/bin/env python3
  
  import sys
  from os.path import dirname, abspath
  sys.dont_write_bytecode=True
  sys.path.append(dirname(dirname(abspath(__file__))))
  from Scripts.Common.VirtualLab import VLSetup

Simulation
**********

This is used to select the 'type' of virtual experiment to be conducted.

Usage:
  Simulation = '$TYPE'

Currently available *'$TYPE'* options:
   | ``Tensile``
   | ``LFA``
   | ``HIVE``

See below for further details.

Mechanical
##########

* `Tensile <virtual_exp.html#tensile-testing>`_: A standard mechanical tensile test where a 'dog-bone' shaped sample is loaded. The load can be applied as a constant force whilst measuring the displacement or as a constant displacement whilst measuring the required load. This provides information about mechanical properties such as Young's elastic modulus.

Thermal
#######

* `LFA <virtual_exp.html#laser-flash-analysis>`_: Laser flash analysis experiment where a disc shaped sample has a short laser pulse incident on one surface, whilst the temperature change is tracked with respect to time on the opposing surface. This is used to measure thermal diffusivity, which is used to calculate thermal conductivity.

Multi-Physics
#############

* `HIVE <virtual_exp.html#hive>`_: "Heat by Induction to Verify Extremes" (HIVE) is an experimental facility at the UK Atomic Energy Authority's (UKAEA) Culham site. It is used to expose plasma-facing components to the high thermal loads they will be subjected to in a fusion energy device. In this experiment, samples are thermally loaded on one surface by induction heating whilst being actively cooled with pressurised water.


Project
*******

User-defined field to specify the name of the project being worked on.

Usage:
  Project = *'$USER_STRING'*

StudyName
*********

User-defined field used to group together virtual experiments.

Usage:
  StudyName = *'$USER_STRING'*

For example, if a parameter sweep of a *'Tensile'* load magnitude was being conducted as part of the *'additive-manufacturing'* project, this could be set to *'LoadMagnitudeSweep'*.

Results from all simulations which are part of this 'study' will be saved in a sub-directory with this name. For this example::

  Output/Tensile/additive-manufacturing/LoadMagnitudeSweep

Parameters_Master
*****************

Name of file which includes values for all the required variables for the selected virtual experiment. These values are used to describe each stage of the particular ‘Study’ to be conducted: pre-processing; simulation; post-processing

This file must be in the directory *'Input/$SIMULATION/$PROJECT'*.

The variables are associated with the python namespaces below:

Mesh
  Variables within this namespace define the parameters required by **SALOME** to construct a mesh, such as geometric dimensions or mesh fineness.

  The file specified by ``Mesh.File`` is executed in **SALOME** using the attributes of the *'Mesh'* namespace to create the geometry which is subsequently meshed.

  The script ``Mesh.File`` must be in the directory *'Scripts/$SIMULATION/Mesh'*.

  Values for each ``Mesh.$VARIABLE_NAME`` are passed to ``Mesh.File``.

Sim
  Variables within this namespace define the parameters needed by **Code_Aster** to perform a FE simulation.

  The file specified by ``Sim.AsterFile`` is executed in **Code_Aster** using the attributes of the *'Sim'* namespace to initiate the simulation.

  The script ``Sim.AsterFile`` must be in the directory *'Scripts/$SIMULATION/Aster'*.

  Values for each ``Sim.$VARIABLE_NAME`` are passed to ``Sim.AsterFile``.

  The *'Sim'* namespace has pre and post-processing options: *'Sim.PreAsterFile'*;  *'Sim.PostAsterFile'*. These are executed before and after the **Code_Aster** call, respectively.

  The scripts for these must be located in *'PreAster'* and *'PostAster'* sub-directories within *'Scripts/$SIMULATION'*.

Usage:
  Parameters_Master = *'$FNAME'*

**NOTE:** Do not include the '.py' file extension as part of $FNAME.

Parameters_Var
**************

This value should be set to the file name (*'$FNAME'*) of the file which includes value ranges for particular variables of the user's choice. These variables must be a sub-set from the full list within *'Parameters_Master'*. These values ranges are used to perform a parameterised 'study' where multiple simulations are conducted concurrently.

Name of file which includes value ranges for particular variables of the user’s choice. These variables must be a subset from the full list within *‘Parameters_Master’*. These values ranges are used to perform a parameterised ‘study’ where multiple simulations are conducted concurrently. 

The specified *'Parameters_Var'* file must be in the directory *'Input/$SIMULATION/$PROJECT'*.

If *'Parameters_Var'* is set to *None* a single simulation is run using the values defined in *Parameters_Master*. When *'Parameters_Var'* is set to *'$FNAME'* those specific values defined in this file will be used instead of those in *Parameters_Master*. If multiple values are given for a single variable, then multiple simulations will be carried out.

Please see the `Tutorials section <examples.html>`_ to see this in action.

Usage:
  Parameters_Var = {*'$FNAME'*/*None*}

**NOTE:** Do not include the '.py' file extension as part of $FNAME.

Mode
****
This dictates how much information is printed in the terminal during the running of VirtualLab. Options available; 'Interactive', 'Continuous', 'Headless'. 'I'/'C'/'H' may be used in place of the full option names.

Usage:
  mode = "$OPTION"

Options:
   | ``Interactive`` Prints all output to the terminal.
   | ``Continuous`` Writes the output to a file as it is generated.
   | ``Headless`` Writes output to file at the end of the process.


VLSetup
*******

The VLSetup class interfaces between the system, **SALOME** and **Code_Aster** to ensure that the full workflow of a virtual experiment can be completed solely via the command line. 

An explanation of the functions of the class and the key-word arguments (kwargs) available are provided below.

Class initiation
################

Firstly the object VirtualLab is created using the VLSetup class. The variables detailed above are passed as arguments, making it possible to differentiate between different virtual
experiments and how results are to be stored. ::

  VirtualLab=VLSetup(Simulation, 
		     Project, 
		     StudyName, 
	 	     Parameters_Master, 
		     Parameters_Var, 
		     Mode, 
		     port=None)

kwargs:
   | ``port`` int (optional)
   |          Specify a port number on which **SALOME** is open. This will save the time required to open & close an instance of **SALOME** in **VirtualLab**. An instance is usually opened on ports starting at 2810. Default is None.

Create
######

This function is responsible for checking that all defined files exist in the expected location. These include Parameters_Master and Parameters_Var and the files specified therein (Mesh.File, Sim.PreAsterFile, Sim.AsterFile, Sim.PostAsterFile). Once this is satisfied, output directories are created for the results, and the necessary files are created to produce mesh(es) and run simulation(s). ::

  VirtualLab.Create(
             RunMesh=True,
             RunSim=True)

kwargs:
   | ``RunMesh`` bool (optional)
   |     Indicates whether or not the meshing routine will be run, which is defined by the 'Mesh' namespace in *Parameters_Master* and *Parameters_Var* . Default is True.
   | ``RunSim``  bool (optional)
   |     Indicates whether or not the simulation routine will be run, which is defined by the 'Sesh' namespace in *Parameters_Master* and *Parameters_Var* . Default is True.


Mesh
####
This function is the meshing routine. The mesh(es) defined using the namespace 'Mesh' in Parameters_Master and Parameters_Var are created and
saved in Output/$SIMULATION/$PROJECT/Meshes along with a file detailing the variables used for their creation. If RunMesh is set to False in 'Create'
then this routine is skipped. This may be useful when different simulation parameters are to be used on a pre-existing mesh. ::

  VirtualLab.Mesh(
           ShowMesh=False,
           MeshCheck=None)

kwargs:
   | ``ShowMesh`` bool (optional)
   |     Indicates whether or not to open created mesh(es) in the **SALOME** GUI for visualisation to assess their suitability. VirtualLab will terminate once the GUI is closed and no simulation will be carried out. Default is False.
   | ``MeshCheck`` '$MESH_NAME' (optional)
   |     Meshes '$MESH_NAME' in the **SALOME** GUI to help with debugging if there are errors. Default is None.

Sim
###

This function is the simulation routine. The simulation(s), defined using the namespace 'Sim' in Parameters_Master and Parameters_Var, are carried out. The results are saved to Output/$SIMULATION/$PROJECT/$STUDYNAME. This routine also runs pre/post processing scripts provided through Sim.PreAsterFile and Sim.PostAsterFile, both of which are optional. If RunSim is set to False in 'Create' then this routine is skipped. ::

  VirtualLab.Sim(
           RunPreAster=True,
           RunAster=True,
           RunPostAster=True,
           ShowRes=False,
           ncpus=1,
           memory=2,
           mpi_nbcpu=1,
           mpi_nbnoeud=1)

kwargs:
   | ``RunPreAster`` bool (optional)
   |     Indicates whether or not to run the optional pre-processing script provided in Sim.PreAsterFile. Default is True.
   | ``RunAster`` bool (optional)
   |     Indicates whether or not to run the **Code_Aster** script provided in Sim.AsterFile. Default is True.
   | ``RunPostAster`` bool (optional)
   |     Indicates whether or not to run the optional post-processing script provided in Sim.PostAsterFile. Default is True.
   | ``ShowRes`` bool (optional)
   |     Visualises the .rmed file(s) produced by **Code_Aster** by opening ParaVis. Default is False.
   | ``ncpus`` int (optional)
   |     Number of processors used by the solver 'MULT_FRONT' in **Code_Aster**. Default is 1.
   | ``memory`` float (optional)
   |     Number of GBs of memory allocated to **Code_Aster** for simulations. Default is 2.
   | ``mpi_nbcpu`` int (optional)
   |     Number of cpus cores for MPI parallelism. Default is 1.
   | ``mpi_nbnoeud`` int (optional)
   |     Number of nodes which mpi_nbnoeud are spread over. Default is 1.

**NOTE:** The binary distribution of standalone **Code_Aster** and the version packaged with **Salome-Meca** does not make use of MPI. To use MPI with **Code_Aster** it must be compiled from source, in which case the solvers 'MUMPS' and 'PETSC' may be used. 

For example, mpi_nbcpu=12,mpi_nbnoeud=4 will set the solver to use 12 cores over 4 nodes, i.e. 3 cores per node. Alternatively, mpi_nbcpu=2,mpi_nbnoeud=2 will use 2 cores over 2 nodes, i.e. one core per node.

ncpus and mpi_nbcpu will not conflict because only one value is used depending on the solver utilised. That is, if both variables are set, only one is passed to the solver.


Cleanup
#######

This function removes all tmp directories created and closes the opened instance of **SALOME**. ::

  VirtualLab.Cleanup()


