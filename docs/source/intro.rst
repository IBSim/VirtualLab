Introduction
============

VirtualLab is a python package which enables the user to run Finite Element (FE) simulations completely via the command line. The pre and post processing is carried out using the software *SALOME*, while the FE solver *CodeAster* is used for the simulation. While this package has been written for command line use some capabilites have been included to use the GUI for debugging and training.

Setup
*****

To use VirtualLab there are a few things which must be set up. The first is that you have both *SALOME* and *CodeAster* installed on your computer. It is advisable to download both of these through the *SalomeMeca* package. Once this is done you need to add the *SALOME* directory to your path so that salome can be opened by just typing ‘salome’ in a terminal window. Talk to Llion

Layout
******

In the top level directory (TLD) of VirtualLab you will find the *Run* file which is used to launch simulations along with the essential directories needed for running simulation; *Scripts*, *Materials* and *Input*. Alongside these a directory named *Output* will be created containing all the information relevant to your analysis. The contents of these are explained in more detail below. 

Run
#####

This file contains the relevant information required to run analysis using VirtualLab, such as the type of simulation or the mode in which it is run.

Simulation
----------

This indicated the type of simulation which will be run. The simulations currently available are:

* Tensile: A standard tensile test where a component can either be loaded with a constant force or constant displacement.

* LFA: Laser flash analysis experiment where a component is pulsed with a short, high heat flux from a laser.

* HIVE: Heat by Induction to Verify Extremes is an experimental facility at the UK Atomic Energy Authority to expose plasma-facing components to the high temperatures they will face in a fusion reactor.

Project
-------

This specifies the project you are working on, such as a type of component which is being tested. 

StudyName
---------

This groups together simulations, for example if you are testing out different magnitude of loads, you could name this ‘LoadingAnalysis’ and all relevant simulations will be saved here.

Parameters_Master
-----------------

This file contains all of the information required to use VirtualLab. Inside each are the python namespaces *Mesh* and *Sim*. *Mesh* defines the parameters required by *SALOME* to construct a mesh, such as geomtrical dimensions and mesh fineness. The file specified at *Mesh.File* is executed in *SALOME* using the attributes of *Mesh* to create an idealised geometry which is then meshed. The script *Mesh.File* must be in the directory *Scripts/$SIMULATION/Mesh*. *Sim* defines the parameters needed by *CodeAster* to perform a FE simulation. The file executed by *CodeAster* is specified at *Sim.AsterFile* and must be in *Scripts/$SIMULATION/Aster*. There is also the option of having *Sim.PreAsterFile* and/or *Sim.PostAsterFile* which are executed before and after the *CodeAster* part respectively for pre/post-processing needs. If used these are located in *PreAster* and *PostAster* sub-directories of *Scripts/$SIMULATION*

Parameters_Var
--------------

This file is used in conjunction with *Parameters_Master* to enable the running of multiple simulations concurrently. If *Parameters_Var* is set to None a single simulation is run using the values defined in *Parameters_Master*, however if it is provided the values defined in this file will be used instead of those in *Parameters_Master*. For a clearer explanation of this please see the tutorials.

The Parameters_Master and Parameters_Var files specified must be in the directory *Input/$SIMULATION/$PROJECT*. 

Mode
----

You will notice that when the VLSetup class is initialised a key-word argument (kwarg) *mode* is provided. This dictates how much information is printed in the terminal during the running of VirtualLab. The options available are;

* Interactive: Prints all output to the terminal

* Continuous: Writes the output to a file as it is generated

* Headless: Writes output to file at the end of the process (default)

Scripts
#######

This directory contains all scripts necessary for setting up and running VirtualLab. The sub-directory *Common* contains the scripts necessary to setup the VirtualLab environment for any type of simulation. There are also sub-directories for each type of simulation, inside which are the relevant *SALOME* and *CodeAster* scripts used, along with additional simulation-specific data, such as different laser pulse profiles for the LFA simulations. 

Materials
#########

This directory contains the material properties which will be used for simulations. The sub-directories are different materials, each of which contain material properties. Some materials are non-linear with temperature dependence. 

Input
#####

As previously outlined the Input directory contains the *Parameters_Master* and *Parameters_Var* files referenced previously and can be found in *Input/$SIMULATION/$PROJECT*. 

Output
######

This directory will be created in the TLD to hold all of the data generated by VirtualLab. The meshes created can be found in *Output/$SIMULATION/$PROJECT/Meshes*, while the simulation results can be found in *Output/$SIMULATION/$PROJECT/$STUDYNAME*


