#!/usr/bin/env python3
import sys
from os.path import dirname, abspath
sys.dont_write_bytecode=True
sys.path.append(dirname(dirname(abspath(__file__))))
from Scripts.Common.VirtualLab import VLSetup

'''
Usage:
    ./Run.py

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
    experiments and how results are to be stored.

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
    StudyName : '$USER_STRING' (str)
        User-defined field used to group together virtual experiments.
    Parameters_Master : ‘$FNAME’ (str)
        Name of file which includes values for all the required variables for
        the selected virtual experiment. These values are used to describe each
        stage of the particular ‘Study’ to be conducted:
            pre-processing; simulation; post-processing
        This file must be in the directory ‘Input/$SIMULATION/$PROJECT’.
    Parameters_Var : {‘$FNAME’/None} (str)
        Name of file which includes value ranges for particular variables of the
        user’s choice. These variables must be a subset from the full list
        within ‘Parameters_Master’. These values ranges are used to perform a
        parameterised ‘study’ where multiple simulations are conducted
        concurrently. If Parameters_Var is set to 'None' a single simulation
        will be run. This file must be in the directory
        ‘Input/$SIMULATION/$PROJECT’.
    Mode : '$TYPE' (str)
        This dictates how much information is printed in the terminal during the
        running of VirtualLab. Options available; 'Interactive', 'Continuous',
        'Headless'. 'I'/'C'/'H' may be used in place of the full option names
          'Intercative' - Prints all output to the terminal.
          'Continuous' - Writes the output to a file as it is generated.
          'Headless' - Writes output to file at the end of the process.
    port : int (optional)
        Specify a port number on which SALOME is open. This will save the time
        required to open & close an instance of SALOME. SALOME is usually opened
        on ports starting at 2810. Default is 'None'.

VirtualLab.Create() :
    This function is responsible for checking that all defined files exist in
    the expected location. These include Parameters_Master and Parameters_Var
    and the files specified therein (Mesh.File, Sim.PreAsterFile, Sim.AsterFile,
    Sim.PostAsterFile). Once this is satisfied, output directories are created
    for the results, and the necessary files are created to produce mesh(es) and
    run simulation(s).
    RunMesh : bool (optional)
        This indicates whether or not the meshing routine will be run, which is
        defined by the 'Mesh' namespace in Parameters_Master. Default is True.
    RunSim : bool (optional)
        This indicates whether or not the simulation routine will be run, which
        is defined by the 'Sim' namespace in Parameters_Master. Default is True.

VirtualLab.Mesh() :
    This function is the meshing routine. The mesh(es) defined using the
    namespace 'Mesh' in Parameters_Master and Parameters_Var are created and
    saved in Output/$SIMULATION/$PROJECT/Meshes along with a file detailing the
    variables used for their creation. If RunMesh is set to False in 'Create'
    then this routine is skipped. This may be useful when different simulation
    parameters are to be used on a pre-existing mesh.
    ShowMesh : bool (optional)
        Indicates whether or not to open created mesh(es) in the SALOME GUI for
        visualisation to assess their suitability. VirtualLab will terminate
        once the GUI is closed and no simulation will be carried out. Default is
        False.
    MeshCheck : {'$MESH_NAME'/None} (str, optional)
        Meshes '$MESH_NAME' in the SALOME GUI to help with debugging if
        there are errors. Default is None.

VirtualLab.Sim() :
    This function is the simulation routine. The simulation(s), defined using
    the namespace 'Sim' in Parameters_Master and Parameters_Var, are carried
    out. The results are saved to Output/$SIMULATION/$PROJECT/$STUDYNAME. This
    routine also runs pre/post processing scripts provided through
    Sim.PreAsterFile and Sim.PostAsterFile, both of which are optional. If
    RunSim is set to False in 'Create' then this routine is skipped.
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
    ncpus : int (optional)
        Number of processors used by the solver 'MULT_FRONT' in Code_Aster.
        Default is 1.
    memory : int (optional)
        Number of GBs of memory allocated to Code_Aster for simulations.

    ### Code acceleration through parallelism with MPI ###
    The binary distribution of standalone Code_Aster and the version packaged
    with Salome-Meca does not make use of MPI. To use MPI with Code_Aster it
    must be compiled from source, in which case the solvers 'MUMPS' and 'PETSC'
    may be used.
    mpi_nbcpu : int (optional)
        Number of cpus cores for MPI parallelism. Default is 1.
    mpi_nbnoeud : int (optional)
        Number of nodes which mpi_nbnoeud are spread over. Default is 1.
    For example, mpi_nbcpu=12,mpi_nbnoeud=4 will set the solver to use 12 cores
    over 4 nodes, i.e. 3 cores per node.
    Alternatively, mpi_nbcpu=2,mpi_nbnoeud=2 will use 2 cores over 2 nodes, i.e.
    one core per node.

    ncpus and mpi_nbcpu will not conflict because only one value is used
    depending on the solver utilised. That is, if both variables are set, only
    one is passed to the solver.

VirtualLab.Cleanup()
    This function removes all tmp directories created and closes the open
    instance of SALOME (if one was opened by VirtualLab).

'''

Simulation='Tensile'
Project='Example'
StudyName='Training'
Parameters_Master='TrainingParameters'
Parameters_Var='Parametric_1'
Mode='Interactive'

VirtualLab=VLSetup(
           Simulation,
           Project,
           StudyName,
           Parameters_Master,
           Parameters_Var,
           Mode,
           port=None)

VirtualLab.Create(
           RunMesh=True,
           RunSim=True)

VirtualLab.Mesh(
           ShowMesh=False,
           MeshCheck=None)

VirtualLab.Sim(
           RunPreAster=True,
           RunAster=True,
           RunPostAster=True,
           ShowRes=False,
           ncpus=1,
           memory=2,
           mpi_nbcpu=1,
           mpi_nbnoeud=1)

VirtualLab.Cleanup()
