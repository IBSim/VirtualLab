#!/usr/bin/env python3
import sys
from os.path import dirname, abspath
sys.dont_write_bytecode=True
sys.path.append(dirname(dirname(abspath(__file__))))
from Scripts.Common.VirtualLab import VLSetup

'''
This is a template of the file required to launch VirtualLab. Provided is an
explanation of the VLSetup class, the variables passed to it and the key-word
arguments (kwargs) available. Below all of the kwargs available are included with
their default values shown.

The VLSetup class interfaces between SALOME and Code_Aster and creates all of
the necessary directories.

Class initiation :
    Firstly we create the object VirtualLab using VLSetup. The variables passed
    as arguments makes it possible to differentiate between different virtual
    experiments and how the results are stored.
    Simulation : '$TYPE'
        The type of virtual experiment to be conducted. Available types are 'Tensile',
        'LFA', and 'HIVE'.
        'Tensile' - A standard mechanical tensile test where a ‘dog-bone’ shaped
                    sample is loaded. The load can be applied as a constant force
                    whilst measuring the displacement or as a constant displacement
                    whilst measuring the required load.
        'LFA' -     Laser flash analysis experiment where a disc shaped sample has a
                    short laser pulse incident on one surface, whilst the temperature
                    change is tracked with respect to time on the opposing surface.
        'HIVE' -    Heat by Induction to Verify Extremes is an experimental facility
                    at the UK Atomic Energy Authority (UKAEA) to expose plasma-facing
                    components to the high temperatures they will face in a fusion
                    reactor. Samples are thermally loaded on by induction heating
                    whilst being actively cooled with pressurised water.
    Project : '$USER_STRING'
        User-defined field to specify the name of the project being worked on.
    StudyName : '$USER_STRING'
        User-defined field used to group together virtual experiments.
    Parameters_Master : ‘$FNAME’
        File which includes values for all the required variables for the selected
        virtual experiment. These values are used to describe each stage of the
        particular ‘Study’ to be conducted: pre-processing; simulation; post-processing.
        This file must be in the directory ‘Input/$SIMULATION/$PROJECT’.
    Parameters_Var : {‘$FNAME’/None}
        File which includes value ranges for particular variables of the user’s choice.
        These variables must be a sub-set from the full list within ‘Parameters_Master’.
        These values ranges are used to perform a parameterised ‘study’ where multiple
        simulations are conducted concurrently.
        If Parameters_Var is set to None a single simulation will be run.
        This file must be in the directory ‘Input/$SIMULATION/$PROJECT’.
    Mode : '$TYPE'
        This dictates how much information is printed in the terminal during the
        running of VirtualLab. Available types are; 'Interactive','Continuous','Headless'.
        'I'/'C'/'H' may be used in place of the full option names
        'Intercative' - Prints all output to the terminal.
        'Continuous' - Writes the output to a file as it is generated.
        'Headless' - Writes output to file at the end of the process.
    port : int (optional)
        Specify a port where SALOME is open on. This will save the time
        required to open & close an instance of SALOME. SALOME is usually opened on
        ports starting at 2810. Default is None.

VirtualLab.Create() :
    This function is responsible for checking that all files defined exist in the
    expected location. These include Parameters_Master and Parameters_Var and
    the files specified therein (Mesh.File, Sim.PreAsterFile, Sim.AsterFile,
    Sim.PostAsterFile). Once this is satsifed output directories are created for
    the results, and the necessary files are created to produce mesh(es) and run
    simulation(s).
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
    variables used for their creation. If RunMesh is set to False in 'Create' then
    this routine is skipped.
    ShowMesh : bool (optional)
        Indicates whether or not to open created mesh(es) in the SALOME GUI to
        look at to assess their suitability. VirtualLab will terminate once the
        GUI is closed. Default is False.
    MeshCheck : {'$MESH_NAME'/None} (optional)
        Meshes '$MESH_NAME' in the SALOME GUI to help with debugging if
        there are errors. Default is None.

VirtualLab.Sim() :
    This function is the simulation routine. The simulation(s) defined using the
    namespace 'Sim' in Parameters_Master and Parameters_Var are run with the results
    saved to Output/$SIMULATION/$PROJECT/$STUDYNAME. This rotune also runs
    pre/post processing scripts provided through Sim.PreAsterFile and
    Sim.PostAsterFile, both of which are optional. If RunSim is set to False in
    'Create' then this routine is skipped.
    RunPreAster : bool (optional)
        Indicates whether or not to run the optional pre-processing script
        provided in Sim.PreAsterFile. default is True.
    RunAster : bool (optional)
        Indicates whether or not to run the Code_Aster script provided in
        Sim.AsterFile. default is True.
    RunPostAster : bool (optional)
        Indicates whether or not to run the optional post-processing script
        provided in Sim.PostAsterFile. default is True.
    ShowRes : bool (optional)
        Opens the .rmed file(s) produced by Code_Aster in ParaVis to look at.
        Default is False
    ncpus : int (optional)
        Number of processors used by the solver 'MULT_FRONT' in Code_Aster.
        Default is 1.
    memory : int (optional)
        Number of GBs of memory allocated to Code_Aster for simulations.
        
    ## MPI parallelism ###
    # MPI doesn't come as standard with Code_Aster and needs to be compiled
    especially. If this is compiled this can be used with solvers 'MUMPS' and
    'PETSC'
    mpi_nbcpu : int (optional)
        Number of cpus for MPI parallelism. Default is 1.
    mpi_nbnoeud : int (optional)
        Number of nodes which mpi_nbnoeud are spread over. Defaut is 1.

VirtualLab.Cleanup()
    This function removes all tmp directories created and closes the instance of
    SALOME (if one was opened by VirtualLab).

'''

Simulation='Tensile'
Project='Example'
StudyName='Training'
Parameters_Master='TrainingParameters'
Parameters_Var='Parametric_1'
Mode='Interactive'

VirtualLab = VLSetup(Simulation,Project,StudyName,Parameters_Master,Parameters_Var,Mode,port=None)

VirtualLab.Create(RunMesh=True, RunSim=True)

VirtualLab.Mesh(ShowMesh=False, MeshCheck=None)

VirtualLab.Sim(RunPreAster=True, RunAster=True, RunPostAster=True, ShowRes=False, mpi_nbcpu=1, mpi_nbnoeud=1, ncpus=1, memory=2)

VirtualLab.Cleanup()
