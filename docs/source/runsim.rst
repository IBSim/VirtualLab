Running a Simulation
====================

The `RunFiles <structure.html#runfiles>`_. directory contains the driver files to launch virtual experiments. 

These files contain the relevant information required to run analysis using **VirtualLab**, such as the type of virtual experiment or the mode in which it is run.

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

* `HIVE <virtual_exp.html#hive>`_: Heat by Induction to Verify Extremes is an experimental facility at the UK Atomic Energy Authority (UKAEA) to expose plasma-facing components to the high temperatures they will face in a fusion reactor. Samples are thermally loaded on by induction heating whilst being actively cooled with pressurised water.

Project
*******

This is a user-defined field to specify the name of the project being worked on, such as the type of component which is being tested.

Usage:
  Project = *'$USER_STRING'*

StudyName
*********

This is a user-defined field used to group together virtual experiments.

Usage:
  StudyName = *'$USER_STRING'*

For example, if a parameter sweep of a *'Tensile'* load magnitude was being conducted as part of the *'additive-manufacturing'* project, this could be set to *'LoadMagnitudeSweep'*.

Results from all simulations which are part of this 'study' will be saved in a sub-directory with this name. For this example::

  Input/Tensile/additive-manufacturing/LoadMagnitudeSweep

Parameters_Master
*****************

This value should be set to the file name (*'$FNAME'*) of the file which includes values for all the required variables for the selected virtual experiment. These values are used to describe each stage of the particular 'Study' to be conducted: pre-processing; simulation; post-processing.

The specified *'Parameters_Master'* file must be in the directory *'Input/$SIMULATION/$PROJECT'*.

The variables are associated with the python namespaces below:

Mesh
  Variables within this namespace define the parameters required by **SALOME** to construct a mesh, such as geometric dimensions or mesh fineness.

  The file specified by ``Mesh.File`` is executed in **SALOME** using the attributes of the *'Mesh'* namespace to create the geometry which is subsequently meshed.

  The script ``Mesh.File`` must be in the directory *'Scripts/$SIMULATION/Mesh'*.

  Values for each ``Mesh.$VARIABLE_NAME`` are passed to ``Mesh.File``.

Sim
  Variables within this namespace define the parameters needed by **CodeAster** to perform a FE simulation.

  The file specified by ``Sim.AsterFile`` is executed in **CodeAster** using the attributes of the *'Sim'* namespace to initiate the simulation.

  The script ``Sim.AsterFile`` must be in the directory *'Scripts/$SIMULATION/Aster'*.

  Values for each ``Sim.$VARIABLE_NAME`` are passed to ``Sim.AsterFile``.

  The *'Sim'* namespace has pre and post-processing options: *'Sim.PreAsterFile'*;  *'Sim.PostAsterFile'*. These are executed before and after the **CodeAster** call, respectively.

  The scripts for these must be located in *'PreAster'* and *'PostAster'* sub-directories within *'Scripts/$SIMULATION'*.

Usage:
  Parameters_Master = *'$FNAME'*

**NOTE:** Do not include the '.py' file extension as part of $FNAME.

Parameters_Var
**************

This value should be set to the file name (*'$FNAME'*) of the file which includes value ranges for particular variables of the user's choice. These variables must be a sub-set from the full list within *'Parameters_Master'*. These values ranges are used to perform a parameterised 'study' where multiple simulations are conducted concurrently.

The specified *'Parameters_Var'* file must be in the directory *'Input/$SIMULATION/$PROJECT'*.

If *'Parameters_Var'* is set to 'None' a single simulation is run using the values defined in *Parameters_Master*. When *'Parameters_Var'* is set to *'$FNAME'* those specific values defined in this file will be used instead of those in *Parameters_Master*. If multiple values are given for a single variable, then multiple simulations will be carried out.

Please see the `Tutorials section <examples.html>`_ to see this in action.

Usage:
  Parameters_Var = {*'$FNAME'*/*'None'*}

**NOTE:** Do not include the '.py' file extension as part of $FNAME.

Mode
****

A key-word argument (kwarg) *'mode'* may be provided when the VLSetup class is initialised. This dictates how much information is printed in the terminal during the running of **VirtualLab**.

Usage:
  mode = "$OPTION"

Options:
   | ``Interactive`` Prints all output to the terminal.
   | ``Continuous`` Writes the output to a file as it is generated.
   | ``Headless`` Writes output to file at the end of the process (default).

**NOTE:** *"I"*/*"C"*/*"H"* may be used in place of the full option names.

OTHER
*****

!!!MORE INFO NEEDED HERE!!!

# Create directories and Parameter files for simulation ::

  VirtualLab.Create()

# Creates meshes ::

  VirtualLab.Mesh()

# Run Pre-Sim calculations, CodeAster and Post-Sim calculations/imaging ::

  VirtualLab.Sim(ShowRes=True)

# Remove tmp folders ::

  VirtualLab.Cleanup()

