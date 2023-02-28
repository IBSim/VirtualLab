Launching VirtualLab
====================

Command Line Interface
**********************

If **VirtualLab** has been installed correctly, the main program will have been added to your system :bash:`<path>`. In this case, it is possible to call **VirtualLab** from the terminal (also known as command line interface, CLI) or a bash script from any location in your system. To facilitate automation, **VirtualLab** has purposefully been designed to run without a graphical user interface (GUI).

.. _usage:

Usage of 'VirtualLab':
::
  
  VirtualLab -f <path>

More options:
   | :bash:`-f <path>` : Where <path> points to the location of the python `RunFiles <../structure.html#runsim/runfile>`_ (this must be either an absolute path or relative to the current working directory).
   | :bash:`-K <Name=Value>`: Overwrite the value specified for variables/keyword arguments specified in the *Run* file.
   | :bash:`-N` : Flag to turn on/off nvidia gpu support.
   | :bash:`--tcp_port`: tcp port to use for server communication. Default is 9000
   | :bash:`--dry-run` : Flag to update containers without running simulations.
   | :bash:`--debug` : print debug messages for networking.
   | :bash:`--test` : Launch a small container to test installation and communication.
   | :bash:`-h` : Display the help menu.

.. note:: The default behaviour is to exit if no :bash:`<path>` is given.

Batch Mode
~~~~~~~~~~

In batch mode, rather than launching the command directly it is normally entered within a script which is sent to a job scheduler (or workload manager). The command is then out in a queue to be executed when the requested resources become available. Singularity is often the platform of choice for shared HPC resources because it can be used without the user needing admin privileges. This is a Singularity example for the `slurm <https://slurm.schedmd.com/>`_ job scheduler on Supercomputing Wales's sunbird system.

**Apptainer** ::

#!/bin/bash --login
#SBATCH --job-name=VirtualLab
#SBATCH --output=logs/VL.out.%J
#SBATCH --error=logs/VL.err.%J
#SBATCH --time=0-01:00
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=6000

'''
Example batch script used to run VirtualLab.  
'''

module purge
module load apptainer/#VersionNumber
module load mpi/#MPIVersion # Optional, only required if one wants to run on multiple nodes (set launcher to mpi if so)
source ~/.VLprofile # make sure VirtualLab/bin is in $PATH

VirtualLab -f <Path/To/File> -K Mode=H NbJobs=4 Launcher=(process/mpi)

Virtual Machines
****************

Once logged into the VM the user is presented with an Ubuntu desktop environment which can be used identically to a native Linux installation. 
That is, with the use of the CLI in a terminal **VirtualLab** may be launched as detailed in :ref:`Usage <usage>`.
