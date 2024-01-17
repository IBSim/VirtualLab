#!/usr/bin/env python3
################################################################################
### HEADER
################################################################################
import sys
import os
import numpy as N
from os.path import dirname, abspath
sys.dont_write_bytecode=True
sys.path.append(dirname(dirname(abspath(__file__))))
from Scripts.Common.VirtualLab import VLSetup

################################################################################
### README
################################################################################
'''
Usage:
    ./Irradiation.py

This is a template of the file required to launch VirtualLab.

Currently, no options can be passed to the script from the command line.
Options must be selected by editing this file directly.

Below is an explanation of:
 - the VLSetup class
 - its input variables (passed from this script)
 - available keyword arguments (kwargs)

The kwargs below in this script are the default values. That is, this is the
default behaviour if no kwargs are given.

The VLSetup class interfaces between the system, SALOME and Code_Aster. Part of
its activity is to create the necessary directories.

Class initiation :
    Firstly the object VirtualLab is created using VLSetup. The variables passed
    as arguments makes it possible to differentiate between different virtual
    experiments, how VirtualLab is run and where results are to be stored.

    Simulation : '$TYPE' (str)
        The type of virtual experiment to be conducted.
        Currently available options:
        'Tensile' - A standard mechanical tensile test where a ‘dog-bone’ shaped
                    sample is loaded. The load can be applied as a constant
                    force whilst measuring the displacement or as a constant
                    displacement whilst measuring the required load.
        'LFA'     - Laser flash analysis experiment where a disc shaped sample
                    has a short laser pulse incident on one surface, whilst the
                    temperature change is tracked with respect to time on the
                    opposing surface.
        'HIVE'    - "Heat by Induction to Verify Extremes" (HIVE) is an
                    experimental facility at the UK Atomic Energy Authority's
                    (UKAEA) Culham site. It is used to expose plasma-facing
                    components to the high thermal loads they will be subjected
                    to in a fusion energy device. In this experiment, samples
                    are thermally loaded on one surface by induction heating
                    whilst being actively cooled with pressurised water.
    Project : '$USER_STRING' (str)
        User-defined field to specify the name of the project being worked on.


VirtualLab.Settings(Mode='Headless', Launcher='Process', NbThreads=1)
    Optional part where VirtualLab settings can be altered.

    Mode : '$TYPE' (str, optional)
        This dictates how much information is printed in the terminal during the
        running of VirtualLab. Options available; 'Interactive', 'Terminal',
        'Continuous' and 'Headless'. 'I'/'T'/'C'/'H' may be used in place of
        the full option names. Default is Headless.
          'Intercative' - Prints all output to pop-up terminals.
          'Terminal' - Prints all information to the terminal that launched
                       VirtualLab.
          'Continuous' - Writes the output to a file as it is generated.
          'Headless' - Writes output to file at the end of the process.
    Launcher : '$TYPE' (str, optional)
        This defines the method used to launch the VirtualLab study.
        Currently available options are:
        'Sequential' - Each operation is run sequentially (no parallelism).
        'Process' - Parallelism for a single node only. (Default)
        'MPI' - Parallelism over multiple nodes.
    NbThreads : '$USER_INTEGER' (int, optional)
        Defines how many of the studies that will run concurrently when using
        either the 'process' or 'MPI' launcher. Default is 1.


VirtualLab.Parameters(Parameters_Master, Parameters_Var, RunMesh=True,
                      RunSim=True, RunDA=True) :
    This function creates the parameter files used by VirtualLab
    and defines information used by Mehs, Sim and DA.
    It is also responsible for checking that all defined files exist in
    the expected location, such as Parameters_Master, Parameters_Var
    and the files specified therein (Mesh.File, Sim.AsterFile etc.).

    Parameters_Master : ‘$FNAME’ (str)
        Name of file which includes values for all the required variables for
        the selected virtual experiment. These values are used to describe each
        stage of the particular ‘Study’ to be conducted: Mesh, Sim and DA.
        This file must be in the directory ‘Input/$SIMULATION/$PROJECT’.
    Parameters_Var : {‘$FNAME’/None} (str)
        Name of file which includes value ranges for particular variables of the
        user’s choice. These values ranges are used to perform a
        parameterised ‘study’ where multiple simulations are conducted.
        If Parameters_Var is set to 'None' a single simulation
        will be run. This file must also be in the directory ‘Input/$SIMULATION/$PROJECT’.
    RunMesh : bool (optional)
        This indicates whether or not the meshing routine will be run, which is
        defined by the 'Mesh' namespace in Parameters_Master. Default is True.
    RunSim : bool (optional)
        This indicates whether or not the simulation routine will be run, which
        is defined by the 'Sim' namespace in Parameters_Master. Default is True.
    RunDA : bool (optional)
        This indicates whether or not the data analysis routine will be run, which
        is defined by the 'DA' namespace in Parameters_Master. Default is True.

VirtualLab.Mesh() :
    This function is the meshing routine. The mesh(es) defined using the
    namespace 'Mesh' in Parameters_Master and Parameters_Var are created and
    saved in Output/$SIMULATION/$PROJECT/Meshes along with a file detailing the
    variables used for their creation. If RunMesh is set to False in
    VirtualLab.Parameters then this routine is skipped. This may be useful
    when different simulation parameters are to be used on a pre-existing mesh.

    ShowMesh : bool (optional)
        Indicates whether or not to open created mesh(es) in the SALOME GUI for
        visualisation to assess their suitability. VirtualLab will terminate
        once the GUI is closed and no simulation will be carried out. Default is
        False.
    MeshCheck : {'$MESH_NAME'/None} (str, optional)
        '$MESH_NAME' is constructed in the SALOME GUI for debugging. Default is
        None.

VirtualLab.Sim() :
    This function is the simulation routine. The simulation(s), defined using
    the namespace 'Sim' in Parameters_Master and Parameters_Var, are carried
    out. The results are saved to Output/$SIMULATION/$PROJECT. This
    routine also runs pre/post processing scripts provided through
    Sim.PreAsterFile and Sim.PostAsterFile, both of which are optional. If
    RunSim is set to False in VirtualLab.Parameters then this routine is skipped.

    RunPreAster : bool (optional)
        Indicates whether or not to run the optional pre-processing script
        provided in Sim.PreAsterFile. Default is True.
    RunAster : bool (optional)
        Indicates whether or not to run the Code_Aster script provided in
        Sim.AsterFile. Default is True.
    RunPostAster : bool (optional)
        Indicates whether or not to run the optional post-processing script
        provided in Sim.PostAsterFile. Default is True.
    ShowRes : bool (optional)
        Visualises the .rmed file(s) produced by Code_Aster by opening ParaVis.
        Default is False.

VirtualLab.DA() :
    This function is the data analysis routine. The analysis, defined using
    the namespace 'DA' in Parameters_Master and Parameters_Var, are carried
    out. The results are saved to Output/$SIMULATION/$PROJECT. If
    RunDA is set to False in VirtualLab.Parameters then this routine is skipped.

VirtualLab.Cleanup()
    This function removes any temporary directories created.

'''


Simulation='RVE'
Project='Tutorials'
Parameters_Master='TrainingParameters_1'
Parameters_Var='Parameters_1'

VirtualLab=VLSetup(
           Simulation,
           Project)

VirtualLab.Settings(
           Mode='T',
           Launcher='Process',
           NbJobs=1)

VirtualLab.Parameters(
           Parameters_Master,
           Parameters_Var,
           RunParamak=False,
           RunOpenmc=False,
           Runmodelib=True,
           RunDPA=False,
           RunMesh=False,
           RunSim=False,
           RunDA=False)

VirtualLab.Paramak()

VirtualLab.modelib()

VirtualLab.Openmc()

VirtualLab.DPA()
VirtualLab.Mesh(
           ShowMesh=False,
           MeshCheck=None)

VirtualLab.Sim(
           RunPreAster=False,
           RunAster=False,
           RunPostAster=False,
           ShowRes=False)




Simulation='RVE'
Project='Tutorials'
Parameters_Master='TrainingParameters'
Parameters_Var='Parameters'

VirtualLab=VLSetup(
           Simulation,
           Project)

VirtualLab.Settings(
           Mode='T',
           Launcher='Process',
           NbJobs=1)

VirtualLab.Parameters(
           Parameters_Master,
           Parameters_Var,
           RunParamak=False,
           RunOpenmc=False,
           Runmodelib=False,
           RunDPA=True,
           RunMesh=False,
           RunSim=False,
           RunDA=False)

VirtualLab.Paramak()

VirtualLab.modelib()

VirtualLab.Openmc()
VirtualLab.DPA()
VirtualLab.Mesh(
           ShowMesh=False,
           MeshCheck=None)

VirtualLab.Sim(
           RunPreAster=False,
           RunAster=False,
           RunPostAster=False,
           ShowRes=False)
           

Simulation='RVE'
Project='Tutorials'
Parameters_Master='TrainingParameters_2'
Parameters_Var='Parameters_2'

VirtualLab=VLSetup(
           Simulation,
           Project)

VirtualLab.Settings(
           Mode='T',
           Launcher='Process',
           NbJobs=1)

VirtualLab.Parameters(
           Parameters_Master,
           Parameters_Var,
           RunParamak=False,
           RunOpenmc=False,
           Runmodelib=False,
           RunDPA=False,
           RunMesh=False,
           RunSim=False,
           RunDA=False)

VirtualLab.Paramak()

VirtualLab.modelib()

VirtualLab.Openmc()
VirtualLab.DPA()
VirtualLab.Mesh(
           ShowMesh=False,
           MeshCheck=None)

VirtualLab.Sim(
           RunPreAster=False,
           RunAster=False,
           RunPostAster=False,
           ShowRes=False)
           


Simulation='RVE'
Project='Tutorials'
Parameters_Master='TrainingParameters_sim'
Parameters_Var='Parameters_sim'

VirtualLab=VLSetup(
           Simulation,
           Project)

VirtualLab.Settings(
           Mode='T',
           Launcher='Process',
           NbJobs=1)

VirtualLab.Parameters(
           Parameters_Master,
           Parameters_Var,
           RunParamak=False,
           RunOpenmc=False,
           Runmodelib=False,
           RunDPA=False,
           RunMesh=False,
           RunSim=True,
           RunDA=False)

VirtualLab.Paramak()

VirtualLab.modelib()

VirtualLab.Openmc()
VirtualLab.DPA()
VirtualLab.Mesh(
           ShowMesh=False,
           MeshCheck=None)

VirtualLab.Sim( RunPreAster=False,
           RunAster=True,
           RunPostAster=False,
           ShowRes=False)


